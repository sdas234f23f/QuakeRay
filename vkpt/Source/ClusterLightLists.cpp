// Copyright (c) 2026 QuakeRay contributors
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//

#include "ClusterLightLists.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

#include "Generated/ShaderCommonC.h"
#include "LightManager.h"
#include "RgException.h"
#include "UserFunction.h"
#include "WorldLights.h"

namespace vkpt
{
namespace
{

// Lights a cluster list can hold.
constexpr uint32_t kMaxPerList = Q2_LIGHT_LIST_MAX_PER_CELL;

// Lights the top-up pass may add to a cluster its own PVS does not reach.
constexpr uint32_t kTopUpSources = 8;

// The top-up pass indexes light origins on a coarse grid so that a cluster looks at the cells
// its bounds expanded by the reach touch instead of at every light of the frame.
constexpr float    kMinGridCell = 16.0f;      // Quake units
constexpr uint32_t kMaxGridCells = 262144;

/* How far a light may drift before the composition counts it as moved, in Quake units. The
   clusters a light holds are a function of where it stands, so a light that moves freely would
   call for a composition on every frame it moves on. A light that stays inside one quantum of
   the origin the lists were built from keeps them instead. */
constexpr float kSourceQuantum = 16.0f;       // Quake units

/* How far past its own reach a light is granted a cluster, so that the lists built for one
   origin are still right for every origin inside a quantum of it: the grant is kept while the
   light is gated out against the reach of the position it stands at now. The margin covers the
   diagonal of a quantum, which is the farthest two origins of one quantum can be apart. */
constexpr float kSourceMargin = 32.0f;        // Quake units

static_assert(kSourceMargin >= kSourceQuantum * 1.7320508f, "the source margin must cover a whole quantum");

constexpr uint8_t kVisUnknown = 0;
constexpr uint8_t kVisDecoded = 1;
constexpr uint8_t kVisMissing = 2;

/* Milliseconds of the steady clock: the composition is the only thing timed here, and it is
   timed against the host's own sampling points, so it has to be the same kind of number. */
double NowMs()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* Identifies the map tables the lists were built from. A reload that changes the leaf count is
   caught by the count alone, but a reload with the same count and another layout would not be,
   so the tables that decide the composition are folded in as well. */
uint64_t MapSignature(const WorldLights &worldLights)
{
    const uint32_t parts[] =
    {
        worldLights.GetClusterCount(),
        worldLights.GetFaceCount(),
        worldLights.GetVisDataSize(),
        worldLights.GetPvsRowBytes(),
    };

    uint64_t signature = 0xcbf29ce484222325ull;

    for (uint32_t part : parts)
    {
        signature ^= part;
        signature *= 0x100000001b3ull;
    }

    return signature;
}

/* The BSP stores a PVS row as runs: a zero byte is followed by the number of further zero bytes,
   anything else carries eight leaf bits. Mod_DecompressVis expands it into the host's rotating
   buffer, but the rows here belong to the map and outlive the call, so each decodes into storage
   of its own. */
void DecompressVis(const uint8_t *pIn, uint8_t *pOut, uint32_t rowBytes)
{
    uint32_t written = 0;

    while (written < rowBytes)
    {
        if (*pIn != 0)
        {
            pOut[written++] = *pIn++;
            continue;
        }

        uint32_t run = pIn[1];
        pIn += 2;

        if (run > rowBytes - written)
        {
            run = rowBytes - written; // a truncated row must not write past its end
        }

        while (run-- > 0)
        {
            pOut[written++] = 0;
        }
    }
}

} // namespace

void ClusterLightLists::Reset()
{
    worldLights = nullptr;
    mapClusterCount = 0;
    numClusters = 0;
    rowBytes = 0;
    mapSignature = 0;

    for (int i = 0; i < 3; i++)
    {
        worldMins[i] = worldMaxs[i] = 0.0f;
    }

    // The tables keep their capacity: a reload usually needs the ones it just had.
    sources.clear();
    incoming.clear();
    listsValid = false;
    topUpReach = 0.0f;
    visRows.clear();
    visState.clear();
    slotUids.clear();
    slotDist2.clear();
    slotSource.clear();
    slotFill.clear();
    slotTopUp.clear();
    slotBits.clear();
    bitsWords = 0;
    clusterDirty.clear();
    dirtyClusters.clear();
    movedIndices.clear();
    frameToSource.clear();
    compositionOrder = false;
    gridHead.clear();
    gridNext.clear();
    offsets.clear();
    list.clear();
    listEntries = 0;
    listGeneration++;
    granted.clear();
    denied.clear();
    prevUidIndex.clear();
    curUidIndex.clear();

    stats = RgClusterLightStats{};
    warnedAboutFullList = false;
}

void ClusterLightLists::SetSources(const WorldLights &worldLightsRef,
                                   const RgClusterLightSourcesUploadInfo &uploadInfo,
                                   LightManager *pLightManager, UserPrint *pUserPrint, uint32_t frameIndex)
{
    const double tStart = NowMs();

    // The flags and the pass timings describe this frame alone: what is left of them on a frame
    // that composed nothing is exactly zero.
    stats.composedFrames = 0;
    stats.reusedFrames = 0;
    stats.visMs = 0.0f;
    stats.topUpMs = 0.0f;
    stats.fillMs = 0.0f;

    const uint32_t clusterCount = worldLightsRef.GetClusterCount();

    // A map reaches the renderer as new tables, and the tables are what the lists are built
    // from: a leaf count or a PVS that differs from the cached ones means a new map.
    if (worldLights != &worldLightsRef || mapClusterCount != clusterCount || mapSignature != MapSignature(worldLightsRef))
    {
        PrepareTables(worldLightsRef);
    }

    if (uploadInfo.numLights > 0 && uploadInfo.pLights == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, "Cluster light sources: no light array");
    }

    incoming.resize(uploadInfo.numLights);

    for (uint32_t i = 0; i < uploadInfo.numLights; i++)
    {
        incoming[i].uid = uploadInfo.pLights[i].uniqueID;
        incoming[i].origin[0] = uploadInfo.pLights[i].origin.data[0];
        incoming[i].origin[1] = uploadInfo.pLights[i].origin.data[1];
        incoming[i].origin[2] = uploadInfo.pLights[i].origin.data[2];
        incoming[i].cluster = uploadInfo.pLights[i].cluster;

        /* The reach a light states for itself, never wider than the reach the top-up pass works
           with: a light that asked for more than that would be granted clusters the pass it has
           to share them with never looks at, which is more than the host asked for. */
        float reach = uploadInfo.pLights[i].reach;

        if (reach < 0.0f)
        {
            reach = 0.0f;
        }
        else if (uploadInfo.topUpReach > 0.0f && reach > uploadInfo.topUpReach)
        {
            reach = uploadInfo.topUpReach;
        }

        incoming[i].reach = reach;
    }

    CountSourceChanges();

    const bool sameSet = stats.addedSources == 0 && stats.removedSources == 0 &&
                         sources.size() == incoming.size();
    const bool sameReach = std::abs(topUpReach - uploadInfo.topUpReach) < 0.001f;
    const bool composeable = listsValid && sameReach && numClusters > 0;

    // The reach is part of what the composition is made of, so it is compared before it is taken.
    topUpReach = uploadInfo.topUpReach;

    if (composeable && sameSet && stats.movedSources == 0)
    {
        /* The lists of the previous composition are a function of the map and the light set, and
           both are the same, so the sources kept their slots and their neighbours: only the
           counters of the frame's own order and the publication are left to do. */
        stats.reusedFrames = 1;
        stats.walkedSources = 0;
        stats.cachedSources = uploadInfo.numLights;

        if (uploadInfo.allowIncremental == 0 || !compositionOrder || !UpdateSourceRecords())
        {
            ReorderStats();
            sources.swap(incoming); // the frame's own set is what the manager and the getters see
            compositionOrder = false;
            // The counters are back on the frame's own light order, so nothing maps them.
            frameToSource.clear();
        }
        /* Else the lists are the ones of the composition and the frame's own lights stand in
           them, so the next frame is compared against the origins these lights are at now
           rather than against the ones the composition was built from. */
    }
    else if (composeable && sameSet && compositionOrder && uploadInfo.allowIncremental != 0 &&
             UpdateMovedSources(worldLightsRef, pUserPrint))
    {
        /* The light set is the one the lists were built for and the only lights that changed are
           lights that moved: their own slots are all a frame of this shape has to look at, and
           every other list of the scene is the one the composition left. */
        stats.reusedFrames = 1;
    }
    else
    {
        sources.swap(incoming); // the composition runs on the frame's own sources
        // The counters are composed in the order of the sources, which is this frame's order,
        // so there is nothing to map them through.
        frameToSource.clear();
        Compose(worldLightsRef, pUserPrint);
        stats.composedFrames = 1;
        compositionOrder = true;
    }

    const double tPublish = NowMs();

    if (numClusters > 0 && pLightManager != nullptr)
    {
        pLightManager->SetClusterLightLists(frameIndex, numClusters, offsets.data(), list.data(), listEntries,
                                            listGeneration);
    }

    stats.publishMs = float(NowMs() - tPublish);
    stats.totalMs = float(NowMs() - tStart);
}

void ClusterLightLists::PrepareTables(const WorldLights &worldLightsRef)
{
    worldLights = &worldLightsRef;
    mapClusterCount = worldLightsRef.GetClusterCount();
    numClusters = std::min(mapClusterCount, uint32_t(Q2_MAX_CLUSTERS));
    rowBytes = worldLightsRef.GetPvsRowBytes();
    mapSignature = MapSignature(worldLightsRef);

    for (int i = 0; i < 3; i++)
    {
        worldMins[i] = std::numeric_limits<float>::max();
        worldMaxs[i] = -std::numeric_limits<float>::max();
    }

    for (uint32_t c = 0; c < numClusters; c++)
    {
        const RgFloat3D &mins = worldLightsRef.GetClusterMin(c);
        const RgFloat3D &maxs = worldLightsRef.GetClusterMax(c);

        for (int a = 0; a < 3; a++)
        {
            worldMins[a] = std::min(worldMins[a], mins.data[a]);
            worldMaxs[a] = std::max(worldMaxs[a], maxs.data[a]);
        }
    }

    if (numClusters == 0)
    {
        for (int i = 0; i < 3; i++)
        {
            worldMins[i] = worldMaxs[i] = 0.0f;
        }
    }

    visRows.assign(size_t(numClusters) * rowBytes, 0);
    visState.assign(numClusters, kVisUnknown);

    const size_t slots = size_t(numClusters) * kMaxPerList;
    slotUids.assign(slots, 0);
    slotDist2.assign(slots, 0.0f);
    slotSource.assign(slots, 0);
    slotFill.assign(numClusters, 0);
    slotTopUp.assign(slots, 0);
    slotBits.assign(std::max<size_t>(1, numClusters), 0);
    bitsWords = 0;

    clusterDirty.assign(numClusters, 0);
    dirtyClusters.clear();
    movedIndices.clear();
    frameToSource.clear();
    compositionOrder = false;

    offsets.assign(numClusters + 1, 0);
    list.assign(slots, 0);
    listEntries = 0;
    listGeneration++; // the tables of the previous map are not the tables of this one

    // The composition that follows must run even when the new map registers the same lights.
    sources.clear();
    listsValid = false;
    topUpReach = 0.0f;
    warnedAboutFullList = false;
    warnedAboutClusterClamp = false;
}

void ClusterLightLists::Compose(const WorldLights &worldLightsRef, UserPrint *pUserPrint)
{
    const uint32_t numSources = uint32_t(sources.size());
    const float    reach = topUpReach;

    /* The lists are indexed with the cluster of the map, and the tables hold Q2_MAX_CLUSTERS of
       them. The host folds the map into that size, so a map with more clusters than this is a
       host that did not, and the clusters past the end get no lists at all. */
    if (mapClusterCount > uint32_t(Q2_MAX_CLUSTERS) && !warnedAboutClusterClamp && pUserPrint != nullptr)
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
                 "RT: the map has %u clusters, the lists only cover %u (Q2_MAX_CLUSTERS); "
                 "the surfaces beyond that are left without lights\n",
                 mapClusterCount, uint32_t(Q2_MAX_CLUSTERS));
        pUserPrint->Print(buffer);
        warnedAboutClusterClamp = true;
    }

    stats.clusters = numClusters;
    stats.sources = numSources;
    stats.unresolved = 0;
    stats.listEntries = 0;
    stats.grants = 0;
    stats.denied = 0;
    stats.topUpGrants = 0;
    stats.reachGated = 0;
    stats.walkedSources = 0;
    stats.cachedSources = 0;
    stats.fullClusters = 0;

    granted.assign(numSources, 0);
    denied.assign(numSources, 0);

    /* A slot holds a light only while that light's own PVS covers the cluster, so a composition
       starts from empty lists: keeping the slots of the previous frame let every cluster grow to
       the limit and stay there. */
    std::fill(slotFill.begin(), slotFill.end(), 0);

    bitsWords = std::max(1u, (numSources + 63) / 64);
    slotBits.assign(size_t(numClusters) * bitsWords, 0);

    const double tVis = NowMs();

    // Pass 1: every light gives itself to every cluster its leaf sees, minus the clusters beyond
    // the reach the light states for itself.
    for (uint32_t li = 0; li < numSources; li++)
    {
        GrantSource(li);
    }

    // Pass 2: a cluster tops itself up with the lights its own PVS hides but that stand close
    // enough to its bounds, so that a light does not stop lighting a wall it is right next to.
    const double tTopUp = NowMs();
    stats.visMs = float(tTopUp - tVis);

    if (numSources > 0 && BuildGrid(worldLightsRef, reach))
    {
        for (uint32_t c = 1; c < numClusters; c++)
        {
            TopUpCluster(worldLightsRef, c, reach);
        }
    }

    for (uint32_t li = 0; li < numSources; li++)
    {
        stats.grants += granted[li];
        stats.denied += denied[li];
    }

    stats.topUpMs = float(NowMs() - tTopUp);

    const double tFill = NowMs();

    FillLists(pUserPrint);

    listsValid = true;
    compositionOrder = true;
    stats.fillMs = float(NowMs() - tFill);
}

/* Pass one for one light: every cluster the leaf the light resolved into sees, minus the ones
   beyond the reach the light states for itself. The composition runs it for every light; a
   frame whose only change is that a light moved runs it for that light alone, and that is the
   whole of pass one for such a frame -- every other light handed out the same clusters on this
   frame as on the one the lists were composed on. */
void ClusterLightLists::GrantSource(uint32_t sourceIndex)
{
    const uint32_t cluster = sources[sourceIndex].cluster;

    if (cluster == RG_CLUSTER_LIGHT_NO_CLUSTER || cluster >= numClusters)
    {
        stats.unresolved++;
        return; // a light that resolved into no leaf hands out nothing
    }

    stats.walkedSources++;
    stats.cachedSources += (visState[cluster] == kVisDecoded) ? 1 : 0;

    const uint8_t *pVis = GetClusterVis(cluster);

    if (pVis == nullptr)
    {
        return; // the leaf has no row of its own: pass 1 has nothing to hand out
    }

    /* What the walk hands out is the leaf's own PVS, and the reach of the light is the one
       thing that narrows it: a light that states a reach covers a few rooms, not whatever
       the leaf happens to see through a doorway. That reach is granted with the margin
       added, so the cluster stays in the list of a light that then drifts inside one source
       quantum of the origin the list was built from. A light that states no reach of its own
       is held to none, and the PVS row is the whole of the rule for it, as it was. */
    const float lightReach = sources[sourceIndex].reach;
    const bool  reachGated = lightReach > 0.0f;
    const float lightGate = lightReach + kSourceMargin;
    const float lightGateSquared = lightGate * lightGate;

    for (uint32_t j = 0; j < rowBytes; j++)
    {
        const uint8_t bits = pVis[j];

        if (bits == 0)
        {
            continue;
        }

        for (uint32_t k = 0; k < 8; k++)
        {
            if ((bits & (1u << k)) == 0)
            {
                continue;
            }

            // The row has one bit per leaf, and bit zero belongs to the leaf the row came
            // from, which is not a cluster a light can be added to.
            const uint32_t c = (j << 3) + k + 1;

            if (c >= numClusters)
            {
                continue;
            }

            if (reachGated && !WithinReach(sources[sourceIndex].origin, c, lightGateSquared))
            {
                stats.reachGated++;
                continue;
            }

            if (AppendSlot(c, sourceIndex, Dist2ToBounds(sources[sourceIndex].origin, c), false))
            {
                granted[sourceIndex]++;
            }
            else
            {
                denied[sourceIndex]++;
            }
        }
    }
}

/* Pass two for one cluster: the lights the cluster's own PVS hides but that stand close enough
   to its bounds, of which it keeps the closest few. The slots the cluster already holds, the
   ones pass one gave it, are left to stand: the pass runs on the light set of the frame, and a
   cluster that is looked at twice on a frame has to come out the same both times. */
void ClusterLightLists::TopUpCluster(const WorldLights &worldLightsRef, uint32_t cluster, float reach)
{
    if (worldLightsRef.IsClusterSolid(cluster))
    {
        return;
    }

    const RgFloat3D &mins = worldLightsRef.GetClusterMin(cluster);
    const RgFloat3D &maxs = worldLightsRef.GetClusterMax(cluster);
    const uint64_t  *pBits = &slotBits[size_t(cluster) * bitsWords];

    uint32_t best[kTopUpSources];
    float    bestDist2[kTopUpSources];
    uint32_t bestCount = 0;

    int cellLo[3];
    int cellHi[3];

    for (uint32_t a = 0; a < 3; a++)
    {
        cellLo[a] = GridAxis(a, mins.data[a] - reach);
        cellHi[a] = GridAxis(a, maxs.data[a] + reach);
    }

    for (int zc = cellLo[2]; zc <= cellHi[2]; zc++)
    for (int yc = cellLo[1]; yc <= cellHi[1]; yc++)
    for (int xc = cellLo[0]; xc <= cellHi[0]; xc++)
    for (int32_t li = gridHead[GridCell(xc, yc, zc)]; li >= 0; li = gridNext[li])
    {
        if ((pBits[li >> 6] & (1ull << (li & 63))) != 0)
        {
            continue; // pass 1 already gave this light a slot here
        }

        /* The pass works with one reach for every light, but a light that stated a reach
           of its own is held to that one here as well: the clusters a light ends up in
           are what its own reach covers, whichever pass puts them in a list. Such a
           reach is never wider than the one of the pass, so the two are not tested one
           after the other. */
        const float lightReach = sources[li].reach;
        const float gate = (lightReach > 0.0f) ? (lightReach + kSourceMargin) : reach;
        const float gateSquared = gate * gate;

        if (!WithinReach(sources[li].origin, cluster, gateSquared))
        {
            stats.reachGated += (lightReach > 0.0f) ? 1 : 0;
            continue;
        }

        const float dist2 = Dist2ToBounds(sources[li].origin, cluster);

        // A small sorted list of the closest candidates: the pass rejects most of them,
        // and only the ones that survive are handed to the cluster.
        if (bestCount < kTopUpSources)
        {
            uint32_t p = bestCount++;

            while (p > 0 && bestDist2[p - 1] > dist2)
            {
                bestDist2[p] = bestDist2[p - 1];
                best[p] = best[p - 1];
                p--;
            }

            bestDist2[p] = dist2;
            best[p] = uint32_t(li);
        }
        else if (dist2 < bestDist2[kTopUpSources - 1])
        {
            uint32_t p = kTopUpSources - 1;

            while (p > 0 && bestDist2[p - 1] > dist2)
            {
                bestDist2[p] = bestDist2[p - 1];
                best[p] = best[p - 1];
                p--;
            }

            bestDist2[p] = dist2;
            best[p] = uint32_t(li);
        }
    }

    for (uint32_t b = 0; b < bestCount; b++)
    {
        const uint32_t li = best[b];

        if (AppendSlot(cluster, li, bestDist2[b], true))
        {
            granted[li]++;
            stats.topUpGrants++;
        }
        else
        {
            denied[li]++;
        }
    }
}

/* Compacts the slots of every cluster into the lists the shaders read, together with the offset
   each cluster starts at: a cluster that loses or gains a slot moves the lists of every cluster
   after it, so the tables have to be laid out again even when one cluster alone changed. */
void ClusterLightLists::FillLists(UserPrint *pUserPrint)
{
    uint32_t total = 0;

    stats.fullClusters = 0;

    for (uint32_t c = 0; c < numClusters; c++)
    {
        offsets[c] = total;
        total += slotFill[c];
        stats.fullClusters += (slotFill[c] >= kMaxPerList) ? 1 : 0;
    }

    offsets[numClusters] = total;
    listEntries = total;
    stats.listEntries = total;

    for (uint32_t c = 0; c < numClusters; c++)
    {
        const uint32_t  fill = slotFill[c];
        const uint64_t *pSrc = &slotUids[size_t(c) * kMaxPerList];
        uint64_t       *pDst = &list[offsets[c]];

        for (uint32_t s = 0; s < fill; s++)
        {
            pDst[s] = pSrc[s];
        }
    }

    if (stats.fullClusters > 0 && !warnedAboutFullList && pUserPrint != nullptr)
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer),
                 "RT: %u clusters reached the %u light limit, farther lights are not sampled "
                 "there (Q2_LIGHT_LIST_MAX_PER_CELL)\n",
                 stats.fullClusters, kMaxPerList);
        pUserPrint->Print(buffer);
        warnedAboutFullList = true;
    }

    /* The words of this composition are not the words of the frame before it, whatever the
       frame before it published. */
    listGeneration++;
}

/* Takes back every slot a light holds, and queues the clusters that held one: a cluster whose
   lights changed has to choose its top-up set again, and the slot to close is the one slot of
   that cluster that names the light. The lists are cluster major, so which clusters hold the
   light is read off the membership set of each cluster one word at a time, and the slot itself
   is then found by walking the fill of the clusters that hold it: a light holds a few hundred
   slots, and this is the only way to find them without keeping a list of them that every
   eviction would have to keep in step. */
void ClusterLightLists::VacateSource(uint32_t sourceIndex)
{
    const uint32_t word = sourceIndex >> 6;
    const uint64_t bit = 1ull << (sourceIndex & 63);

    for (uint32_t c = 0; c < numClusters; c++)
    {
        uint64_t *pBits = &slotBits[size_t(c) * bitsWords];

        if ((pBits[word] & bit) == 0)
        {
            continue;
        }

        const uint32_t base = c * kMaxPerList;
        uint32_t       fill = slotFill[c];

        for (uint32_t s = 0; s < fill; s++)
        {
            if (slotSource[base + s] != sourceIndex)
            {
                continue;
            }

            /* A list is read as a set, so the slot is closed by moving the last slot of the
               cluster into it, which costs the same whatever the fill is. */
            fill--;

            if (s != fill)
            {
                slotUids[base + s] = slotUids[base + fill];
                slotDist2[base + s] = slotDist2[base + fill];
                slotSource[base + s] = slotSource[base + fill];
                slotTopUp[base + s] = slotTopUp[base + fill];
            }

            break;
        }

        slotFill[c] = fill;
        pBits[word] &= ~bit;

        // The cluster lost one of the lights its top-up set was chosen against, so it is one of
        // the clusters that have to choose again.
        MarkDirty(c);
    }
}

/* Takes the top-up slots of a cluster away and leaves the ones pass one gave it. The set the
   top-up pass keeps is chosen against the lights the cluster holds, so a cluster that lost or
   can gain one has to let go of the choice made against the lights it had, and the old choice
   would otherwise stand in the way of the new one: a light taken twice would take two slots. */
void ClusterLightLists::DropTopUpSlots(uint32_t cluster)
{
    const uint32_t base = cluster * kMaxPerList;
    const uint32_t fill = slotFill[cluster];
    uint32_t       kept = 0;

    for (uint32_t s = 0; s < fill; s++)
    {
        if (slotTopUp[base + s] == 0)
        {
            if (kept != s)
            {
                slotUids[base + kept] = slotUids[base + s];
                slotDist2[base + kept] = slotDist2[base + s];
                slotSource[base + kept] = slotSource[base + s];
            }

            kept++;
            continue;
        }

        // A cluster that no longer holds a light must stop saying that it does, or the top-up
        // pass would skip the light as one pass one had placed.
        const uint32_t li = slotSource[base + s];

        slotBits[size_t(cluster) * bitsWords + (li >> 6)] &= ~(1ull << (li & 63));
    }

    slotFill[cluster] = kept;
}

/* Queues a cluster whose top-up set has to be looked at again on this frame. */
void ClusterLightLists::MarkDirty(uint32_t cluster)
{
    if (cluster >= clusterDirty.size() || clusterDirty[cluster] != 0)
    {
        return;
    }

    clusterDirty[cluster] = 1;
    dirtyClusters.push_back(cluster);
}

/* Takes the origins, the leaves and the reaches this frame registered over the sources the
   lists were composed from, in the place each of them already stands in. Where a light stands
   is what the frame is compared against, so a frame that hands the lists on to the next one
   has to leave the origins of the lights it was given behind in them. */
bool ClusterLightLists::UpdateSourceRecords()
{
    if (frameToSource.size() != incoming.size())
    {
        return false;
    }

    for (uint32_t j = 0; j < uint32_t(incoming.size()); j++)
    {
        const uint32_t i = frameToSource[j];

        if (i >= sources.size())
        {
            return false;
        }

        sources[i].origin[0] = incoming[j].origin[0];
        sources[i].origin[1] = incoming[j].origin[1];
        sources[i].origin[2] = incoming[j].origin[2];
        sources[i].cluster = incoming[j].cluster;
        sources[i].reach = incoming[j].reach;
    }

    return true;
}

/* The lights that moved are the only ones whose own lists changed, so the frame takes their
   slots back, hands them out again from where those lights stand now, and lets the clusters
   that lost a slot, or that a moved light can now reach, choose their top-up set again from the
   lights they hold. Every other slot of the scene is the one the composition left, and it is
   still right: a light that stayed inside one source quantum of the origin its slots were
   granted from stands inside the margin the grant was made with.

   Returns false for a frame this cannot be done for, which is a frame the caller composes. */
bool ClusterLightLists::UpdateMovedSources(const WorldLights &worldLightsRef, UserPrint *pUserPrint)
{
    if (movedIndices.empty() || sources.size() != incoming.size() || granted.size() != sources.size() ||
        frameToSource.size() != incoming.size())
    {
        return false;
    }

    // The lights stand where this frame registered them, and the order of the sources is left
    // alone: every slot names its light by its place in it.
    if (!UpdateSourceRecords())
    {
        return false;
    }

    const float  reach = topUpReach;
    const double tVis = NowMs();

    stats.walkedSources = 0;
    stats.cachedSources = 0;
    stats.topUpGrants = 0;
    stats.reachGated = 0;

    std::fill(clusterDirty.begin(), clusterDirty.end(), 0);
    dirtyClusters.clear();

    /* Pass one for the lights that moved: the slots they held are the ones that stopped being
       right, so they are taken back first and the light is given the clusters its new leaf and
       its own reach cover now. */
    for (uint32_t m = 0; m < uint32_t(movedIndices.size()); m++)
    {
        const uint32_t li = movedIndices[m];

        if (li >= uint32_t(sources.size()))
        {
            continue;
        }

        granted[li] = 0;
        denied[li] = 0;

        VacateSource(li);
        GrantSource(li);
    }

    const double tTopUp = NowMs();
    stats.visMs = float(tTopUp - tVis);

    /* A cluster can gain a moved light in one of two ways: pass one has just handed it out to
       every cluster the light's leaf sees, and the top-up pass can hand it to every cluster
       within the gate of where the light stands now. The second set is what decides which
       clusters have to look at their top-up set again; the first one is the set pass one wrote
       to, and the ones that lost a slot to the move queued themselves as they did. */
    for (uint32_t m = 0; m < uint32_t(movedIndices.size()); m++)
    {
        const uint32_t li = movedIndices[m];

        if (li >= uint32_t(sources.size()))
        {
            continue;
        }

        if (sources[li].cluster == RG_CLUSTER_LIGHT_NO_CLUSTER || sources[li].cluster >= numClusters)
        {
            continue; // a light that resolved into no leaf takes part in neither pass
        }

        const float lightReach = sources[li].reach;
        const float gate = ((lightReach > 0.0f) ? lightReach : reach) + kSourceMargin;
        const float gateSquared = gate * gate;

        for (uint32_t c = 1; c < numClusters; c++)
        {
            if (WithinReach(sources[li].origin, c, gateSquared))
            {
                MarkDirty(c);
            }
        }
    }

    if (!dirtyClusters.empty() && BuildGrid(worldLightsRef, reach))
    {
        for (uint32_t d = 0; d < uint32_t(dirtyClusters.size()); d++)
        {
            DropTopUpSlots(dirtyClusters[d]);
            TopUpCluster(worldLightsRef, dirtyClusters[d], reach);
        }
    }

    stats.topUpMs = float(NowMs() - tTopUp);

    stats.unresolved = 0;
    stats.grants = 0;
    stats.denied = 0;

    for (uint32_t li = 0; li < uint32_t(sources.size()); li++)
    {
        if (sources[li].cluster == RG_CLUSTER_LIGHT_NO_CLUSTER || sources[li].cluster >= numClusters)
        {
            stats.unresolved++;
        }

        stats.grants += granted[li];
        stats.denied += denied[li];
    }

    const double tFill = NowMs();

    FillLists(pUserPrint);

    stats.fillMs = float(NowMs() - tFill);
    stats.clusters = numClusters;
    stats.sources = uint32_t(sources.size());

    return true;
}

/* Appends a light to the list of a cluster, evicting the farthest light when the list is full
   and the newcomer stands closer. The caller measures the distance, so that a light it has
   already turned down does not pay for the measure again. fromTopUp says which pass the light
   comes from, so that a cluster which has to choose again can let go of the lights it took that
   way and keep the ones its own PVS gave it. Returns false when the cluster keeps the lights it
   had. */
bool ClusterLightLists::AppendSlot(uint32_t cluster, uint32_t sourceIndex, float dist2, bool fromTopUp)
{
    const uint32_t  base = cluster * kMaxPerList;
    uint64_t       *pUids = &slotUids[base];
    float          *pDist2 = &slotDist2[base];
    uint32_t       *pSource = &slotSource[base];
    uint8_t        *pTopUp = &slotTopUp[base];
    uint64_t       *pBits = &slotBits[size_t(cluster) * bitsWords];
    const uint32_t  fill = slotFill[cluster];

    if (fill < kMaxPerList)
    {
        pUids[fill] = sources[sourceIndex].uid;
        pDist2[fill] = dist2;
        pSource[fill] = sourceIndex;
        pTopUp[fill] = fromTopUp ? 1 : 0;
        pBits[sourceIndex >> 6] |= 1ull << (sourceIndex & 63);
        slotFill[cluster] = fill + 1;
        return true;
    }

    uint32_t farthest = 0;
    float    farthestDist2 = pDist2[0];

    for (uint32_t s = 1; s < kMaxPerList; s++)
    {
        if (pDist2[s] > farthestDist2)
        {
            farthestDist2 = pDist2[s];
            farthest = s;
        }
    }

    if (dist2 >= farthestDist2)
    {
        return false; // the cluster already holds kMaxPerList closer lights
    }

    // The evicted light is no longer sampled by this cluster, so its bit must stop claiming the
    // opposite: otherwise the top-up pass would skip it as one that pass 1 had already placed.
    const uint32_t evicted = pSource[farthest];
    pBits[evicted >> 6] &= ~(1ull << (evicted & 63));
    pBits[sourceIndex >> 6] |= 1ull << (sourceIndex & 63);

    pUids[farthest] = sources[sourceIndex].uid;
    pDist2[farthest] = dist2;
    pSource[farthest] = sourceIndex;
    pTopUp[farthest] = fromTopUp ? 1 : 0;

    return true;
}

/* Buckets the origins of the resolved lights into a uniform grid over the map, freed of the
   lights that resolved into no leaf. Returns false when there is nothing to index. */
bool ClusterLightLists::BuildGrid(const WorldLights &worldLightsRef, float reach)
{
    if (sources.empty())
    {
        return false;
    }

    float cell = (reach > kMinGridCell) ? reach : kMinGridCell;
    int   dims[3] = { 1, 1, 1 };
    float extent[3];

    for (int a = 0; a < 3; a++)
    {
        extent[a] = std::max(0.0f, worldMaxs[a] - worldMins[a]);
    }

    // Doubling the cell keeps the grid bounded on a map whose bounds dwarf the reach.
    for (;;)
    {
        int64_t cells = 1;

        for (int a = 0; a < 3; a++)
        {
            dims[a] = int(extent[a] / cell) + 1;
            if (dims[a] < 1)
            {
                dims[a] = 1;
            }

            cells *= dims[a];
        }

        if (cells <= int64_t(kMaxGridCells))
        {
            break;
        }

        cell *= 2.0f;
    }

    const int cellCount = dims[0] * dims[1] * dims[2];

    gridCell = cell;
    for (int a = 0; a < 3; a++)
    {
        gridDims[a] = dims[a];
        gridMins[a] = worldMins[a];
    }

    gridHead.assign(cellCount, -1);
    gridNext.assign(sources.size(), -1);

    for (uint32_t li = 0; li < uint32_t(sources.size()); li++)
    {
        if (sources[li].cluster == RG_CLUSTER_LIGHT_NO_CLUSTER || sources[li].cluster >= numClusters)
        {
            continue; // a light with no cluster has no slot to take part in the top-up pass with
        }

        const int ci = GridCell(GridAxis(0, sources[li].origin[0]), GridAxis(1, sources[li].origin[1]),
                                GridAxis(2, sources[li].origin[2]));

        gridNext[li] = gridHead[ci];
        gridHead[ci] = int32_t(li);
    }

    return true;
}

int ClusterLightLists::GridAxis(uint32_t axis, float value) const
{
    int i = int(std::floor((value - gridMins[axis]) / gridCell));

    if (i < 0)
    {
        i = 0;
    }
    else if (i >= gridDims[axis])
    {
        i = gridDims[axis] - 1;
    }

    return i;
}

int ClusterLightLists::GridCell(int x, int y, int z) const
{
    return (z * gridDims[1] + y) * gridDims[0] + x;
}

const uint8_t *ClusterLightLists::GetClusterVis(uint32_t cluster)
{
    if (cluster >= numClusters)
    {
        return nullptr;
    }

    if (visState[cluster] == kVisUnknown)
    {
        const uint8_t *pCompressed = worldLights->GetClusterVis(cluster);

        if (pCompressed == nullptr)
        {
            visState[cluster] = kVisMissing;
        }
        else
        {
            DecompressVis(pCompressed, &visRows[size_t(cluster) * rowBytes], rowBytes);
            visState[cluster] = kVisDecoded;
        }
    }

    return (visState[cluster] == kVisDecoded) ? &visRows[size_t(cluster) * rowBytes] : nullptr;
}

float ClusterLightLists::Dist2ToBounds(const float *pOrigin, uint32_t cluster) const
{
    const RgFloat3D &mins = worldLights->GetClusterMin(cluster);
    const RgFloat3D &maxs = worldLights->GetClusterMax(cluster);

    float dist2 = 0.0f;

    for (int a = 0; a < 3; a++)
    {
        float d = 0.0f;

        if (pOrigin[a] < mins.data[a])
        {
            d = mins.data[a] - pOrigin[a];
        }
        else if (pOrigin[a] > maxs.data[a])
        {
            d = pOrigin[a] - maxs.data[a];
        }

        dist2 += d * d;
    }

    return dist2;
}

/* Same measure as Dist2ToBounds, but gives up as soon as the running sum passes the reach:
   most (light, cluster) pairs of the top-up pass are rejections, and they are rejected on the
   first axis often enough to matter. */
bool ClusterLightLists::WithinReach(const float *pOrigin, uint32_t cluster, float reachSquared) const
{
    const RgFloat3D &mins = worldLights->GetClusterMin(cluster);
    const RgFloat3D &maxs = worldLights->GetClusterMax(cluster);

    float dist2 = 0.0f;

    for (int a = 0; a < 3; a++)
    {
        float d = 0.0f;

        if (pOrigin[a] < mins.data[a])
        {
            d = mins.data[a] - pOrigin[a];
        }
        else if (pOrigin[a] > maxs.data[a])
        {
            d = pOrigin[a] - maxs.data[a];
        }

        dist2 += d * d;

        if (dist2 > reachSquared)
        {
            return false;
        }
    }

    return true;
}

/* Order-independent comparison of the frame against the composition: a camera turn only
   reshuffles the order in which the visible lights are registered, which is not a new light
   set, and must not be paid for with a composition. What is a change is a light that appeared,
   a light that disappeared, a light that had no leaf and gained one or lost the one it had,
   and a light that moved far enough that the reach it stated for itself no longer describes
   where it stands.

   The leaf a light resolved into is not what the lists are made of, and taking it for a change
   is what made them churn: a light that crossed a leaf boundary re-ran the whole map -- every
   light, every PVS row -- while the lists it came out with were the ones of the frame before
   but for its own slots. The leaf is still worth watching for the one thing pass one makes of
   it, and pass one hands out nothing at all for a light with no leaf: gaining a leaf is a light
   reaching the scene, losing one is a light leaving it. */
void ClusterLightLists::CountSourceChanges()
{
    prevUidIndex.clear();
    curUidIndex.clear();

    prevUidIndex.reserve(sources.size());
    curUidIndex.reserve(incoming.size());

    for (uint32_t i = 0; i < uint32_t(sources.size()); i++)
    {
        prevUidIndex.emplace_back(sources[i].uid, i);
    }

    for (uint32_t i = 0; i < uint32_t(incoming.size()); i++)
    {
        curUidIndex.emplace_back(incoming[i].uid, i);
    }

    std::sort(prevUidIndex.begin(), prevUidIndex.end());
    std::sort(curUidIndex.begin(), curUidIndex.end());

    stats.addedSources = 0;
    stats.removedSources = 0;
    stats.movedSources = 0;

    movedIndices.clear();
    frameToSource.assign(incoming.size(), 0);

    size_t prev = 0;
    size_t cur = 0;

    while (prev < prevUidIndex.size() || cur < curUidIndex.size())
    {
        if (cur >= curUidIndex.size() ||
            (prev < prevUidIndex.size() && prevUidIndex[prev].first < curUidIndex[cur].first))
        {
            stats.removedSources++;
            prev++;
            continue;
        }

        if (prev >= prevUidIndex.size() || curUidIndex[cur].first < prevUidIndex[prev].first)
        {
            stats.addedSources++;
            cur++;
            continue;
        }

        const Source &prevSource = sources[prevUidIndex[prev].second];
        const Source &curSource = incoming[curUidIndex[cur].second];

        // Where the light of the frame stands among the sources the lists were composed from.
        frameToSource[curUidIndex[cur].second] = prevUidIndex[prev].second;

        const bool prevPlaced = prevSource.cluster != RG_CLUSTER_LIGHT_NO_CLUSTER;
        const bool curPlaced = curSource.cluster != RG_CLUSTER_LIGHT_NO_CLUSTER;

        if (prevPlaced != curPlaced)
        {
            // A light that gained a leaf reaches the scene and one that lost it leaves it: both
            // are a light whose slots are not the ones the composition gave it.
            stats.movedSources++;
            movedIndices.push_back(prevUidIndex[prev].second);
        }
        else if (prevPlaced)
        {
            float drift = 0.0f;

            for (int a = 0; a < 3; a++)
            {
                drift = std::max(drift, std::abs(prevSource.origin[a] - curSource.origin[a]));
            }

            // The quantum is what keeps a light that walks from re-running the map on every
            // frame of the walk, and the margin the passes grant the light with is what keeps
            // the lists right while it stays inside one.
            if (drift > kSourceQuantum || std::abs(prevSource.reach - curSource.reach) > 0.5f)
            {
                stats.movedSources++;
                movedIndices.push_back(prevUidIndex[prev].second);
            }
        }

        prev++;
        cur++;
    }
}

/* The counters of the last composition are indexed by the order it was composed in, while the
   frame registers the same lights in the order the visible lists hand them over. Matching the
   two uid orders puts the counters back on the frame's own light index. */
void ClusterLightLists::ReorderStats()
{
    std::vector<uint32_t> newGranted(granted.size(), 0);
    std::vector<uint32_t> newDenied(denied.size(), 0);

    size_t prev = 0;
    size_t cur = 0;

    while (prev < prevUidIndex.size() && cur < curUidIndex.size())
    {
        if (prevUidIndex[prev].first < curUidIndex[cur].first)
        {
            prev++;
            continue;
        }

        if (curUidIndex[cur].first < prevUidIndex[prev].first)
        {
            cur++;
            continue;
        }

        const uint32_t from = prevUidIndex[prev].second;
        const uint32_t to = curUidIndex[cur].second;

        if (from < granted.size() && to < newGranted.size())
        {
            newGranted[to] = granted[from];
            newDenied[to] = denied[from];
        }

        prev++;
        cur++;
    }

    granted.swap(newGranted);
    denied.swap(newDenied);
}

/* The counters are kept in the order the lists stand in. A frame that was handed the lists of
   the previous one keeps the lights where they were composed, and the caller asks for its own
   light order: the map this frame's lights stand in says which counter belongs to which of
   them, and it is empty on a frame that put the counters back on the frame's own order. */
void ClusterLightLists::GetGrants(uint32_t *pGranted, uint32_t *pDenied, uint32_t maxCount, uint32_t *pCount) const
{
    const uint32_t count = std::min(maxCount, uint32_t(granted.size()));

    for (uint32_t i = 0; i < count; i++)
    {
        uint32_t from = i;

        if (i < frameToSource.size())
        {
            from = frameToSource[i];

            if (from >= granted.size())
            {
                continue;
            }
        }

        if (pGranted != nullptr)
        {
            pGranted[i] = granted[from];
        }

        if (pDenied != nullptr)
        {
            pDenied[i] = denied[from];
        }
    }

    if (pCount != nullptr)
    {
        *pCount = uint32_t(granted.size());
    }
}

void ClusterLightLists::GetClusterList(uint32_t cluster, uint64_t *pUniqueIds, uint32_t maxCount, uint32_t *pCount) const
{
    if (cluster >= numClusters)
    {
        if (pCount != nullptr)
        {
            *pCount = 0;
        }

        return;
    }

    const uint32_t count = std::min(maxCount, slotFill[cluster]);
    const uint64_t *pSrc = &slotUids[size_t(cluster) * kMaxPerList];

    for (uint32_t i = 0; i < count; i++)
    {
        if (pUniqueIds != nullptr)
        {
            pUniqueIds[i] = pSrc[i];
        }
    }

    if (pCount != nullptr)
    {
        *pCount = slotFill[cluster];
    }
}

} // namespace vkpt
