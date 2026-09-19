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

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <vkpt/vkpt.h>

#include "Common.h"

namespace vkpt
{

class LightManager;
class UserPrint;
class WorldLights;

// Composes the per-cluster light lists the light manager hands to the shaders. The host only
// registers which lights exist, where each of them stands, and how far each one reaches;
// everything else the composition is made of -- the map PVS, the cluster bounds -- comes from
// WorldLights, so the lists are a function of the map and the registered light set alone.
//
// That is what makes the composition reusable: while the light set is the one the lists were
// composed from and no light moves far enough that the reach it stated for itself no longer
// describes where it stands, the lists of the previous frame are still exactly right, and the
// frame only pays for handing them to the light manager again. A frame that does compose runs
// the same two passes the host used to run: one light reaches every cluster its leaf's PVS
// marks visible and that its own reach covers, a cluster keeps the closest of them in a fixed
// number of slots, and a second pass tops a cluster up with the lights its own PVS hides but
// that the top-up reach brings close enough.
//
// Between the two stands the frame in which the light set is unchanged but for lights that
// moved: a light that stays inside one source quantum of the origin its slots were granted
// from left its lists right, and the margin its reach was granted with is what lets the frame
// show the moved light where it stands now without walking a single other PVS row.
//
// The frame whose light set itself changed is the third shape, and it is not one the whole map
// has to be composed for either: a light that appeared, a light that disappeared or one that
// moved is a handful out of the set, and the lists the composition left are wrong only where
// those lights reach. A light the frame still holds keeps the place it holds in the sources, and
// with that place every slot that names it; the place a light that disappeared held is kept as a
// tombstone, which holds no slot and is the place the next light that appears is given, so that
// no slot of any other light moves and a set that churns does not grow; and a light that appears
// is granted the clusters it reaches from where it stands. Only the clusters the changes reach are
// composed again, and the composition stays the answer wherever that cannot be proved right, which
// is also what it does with the tombstones: it compacts them.
class ClusterLightLists
{
public:
    ClusterLightLists() = default;

    // Registers the lights of the frame, composes the lists when the light set changed, and
    // always publishes them to the light manager.
    void SetSources(const WorldLights &worldLights, const RgClusterLightSourcesUploadInfo &uploadInfo,
                    LightManager *pLightManager, UserPrint *pUserPrint, uint32_t frameIndex);

    // Forgets the map together with the lists: the next SetSources composes from scratch.
    void Reset();

    const RgClusterLightStats &GetStats() const { return stats; }

    // Slot accounting of the lights of the last frame, in the order the frame registered them,
    // and one cluster list as the lists were left for it.
    void GetGrants(uint32_t *pGranted, uint32_t *pDenied, uint32_t maxCount, uint32_t *pCount) const;
    void GetClusterList(uint32_t cluster, uint64_t *pUniqueIds, uint32_t maxCount, uint32_t *pCount) const;

private:
    // One registered light: what it is, where it stands, how far it reaches, and the leaf that
    // resolved it. They are kept together so that a frame can be compared against the
    // composition without touching the map again.
    struct Source
    {
        uint64_t uid;
        float    origin[3];
        uint32_t cluster;
        // Distance up to which the light belongs in a list, zero for no limit of its own. It
        // is clamped to the top-up reach when the sources are taken: the two passes have to
        // agree on where a light stops mattering.
        float    reach;
        /* Set on a place a light left behind, which is kept as a place that holds no light and
           that no pass reads, so that the lights that stayed keep the places their slots name
           them by. It is the place the next light that appears is given, and no slot is made of
           it while it holds none. */
        bool     tombstone;
    };

    void PrepareTables(const WorldLights &worldLights);
    void Compose(const WorldLights &worldLights, UserPrint *pUserPrint);
    // Pass one for one light, as the composition runs it for every light.
    void GrantSource(uint32_t sourceIndex);
    // Pass two for one cluster, as the composition runs it for every cluster.
    void TopUpCluster(const WorldLights &worldLights, uint32_t cluster, float reach);
    // Takes back the slots a light holds and the top-up slots of every cluster that holds one
    // of them, so that a light that changed can be placed again without composing the lists of
    // a cluster whose own lights did not change.
    void VacateSource(uint32_t sourceIndex);
    void DropTopUpSlots(uint32_t cluster);
    void MarkDirty(uint32_t cluster);
    void FillLists(UserPrint *pUserPrint);
    // Places the lights that changed -- the ones that moved, the ones that appeared and the ones
    // that disappeared -- and returns false when the frame is not one this can be done for and
    // the caller has to compose.
    bool UpdateSourceSet(const WorldLights &worldLights, UserPrint *pUserPrint);
    // Lays the membership set out again with a new stride, keeping the bit of every source where
    // the place it names stands. The stride grows only when the set of places does.
    void ResizeSlotBits(uint32_t newWords);
    // Takes the origins, the leaves and the reaches of this frame's lights over the sources the
    // lists were composed from, leaving their order alone. False when the frame is not made of
    // the lights the lists hold.
    bool UpdateSourceRecords();
    bool AppendSlot(uint32_t cluster, uint32_t sourceIndex, float dist2, bool fromTopUp);
    bool BuildGrid(const WorldLights &worldLights, float reach);
    int  GridAxis(uint32_t axis, float value) const;
    int  GridCell(int x, int y, int z) const;
    const uint8_t *GetClusterVis(uint32_t cluster);
    float Dist2ToBounds(const float *pOrigin, uint32_t cluster) const;
    bool  WithinReach(const float *pOrigin, uint32_t cluster, float reachSquared) const;
    void  CountSourceChanges();
    void  ReorderStats();

    const WorldLights *worldLights = nullptr;
    // Leaf count of the map the tables were prepared for, before clamping to the cluster cap.
    uint32_t mapClusterCount = 0;
    // Number of clusters the lists have room for: the leaf count, clamped to what the shaders
    // and the light list buffers are sized for.
    uint32_t numClusters = 0;
    uint32_t rowBytes = 0;
    // Identifies the map tables the tables above were prepared for, so that a reload is
    // noticed even when the new map has the same leaf count.
    uint64_t mapSignature = 0;

    float worldMins[3] = {};
    float worldMaxs[3] = {};

    // The sources of the composition and the sources of the frame being set.
    std::vector<Source> sources;
    std::vector<Source> incoming;
    bool  listsValid = false;
    float topUpReach = 0.0f;

    // Decoded PVS row of every cluster a light resolved into, kept for the lifetime of the map:
    // a row is reached by many lights, and decoding it per light per frame was the largest
    // single cost of the composition.
    std::vector<uint8_t> visRows;
    std::vector<uint8_t> visState;

    std::vector<uint64_t> slotUids;   // kMaxPerList per cluster
    std::vector<float>    slotDist2;
    std::vector<uint32_t> slotSource;
    std::vector<uint32_t> slotFill;   // one per cluster
    // Set while a slot holds a light the top-up pass put there rather than the PVS walk, so
    // that a cluster which loses or can gain a light can let go of the choice that was made
    // against the lights it had.
    std::vector<uint8_t>  slotTopUp;  // kMaxPerList per cluster
    // Membership of a cluster, one bit per source: set while the source holds a slot there, so
    // that the top-up pass does not hand the same light to the same cluster twice.
    std::vector<uint64_t> slotBits;
    uint32_t              bitsWords = 0;
    // Clusters whose top-up set has to be looked at again on this frame, and the lights that
    // changed on it. All of them are left over between frames only as capacity.
    std::vector<uint8_t>  clusterDirty;   // one per cluster
    std::vector<uint32_t> dirtyClusters;
    std::vector<uint32_t> movedIndices;   // places in the sources
    std::vector<uint32_t> addedIndices;   // lights of the frame
    std::vector<uint32_t> removedIndices; // places in the sources
    std::vector<uint32_t> changedIndices; // places in the sources, moved and added together
    /* Places of the sources whose light left the scene, which hold no slot and are granted to no
       other light for as long as they are kept: every slot names its light by the place it holds,
       and a place that is kept names a light that is gone. They are kept because the lights that
       stay keep the places their slots name them by, and they are given back to the lights that
       appear, so that a set that churns neither grows nor has to be composed; the set is
       compacted by a composition once they outgrow the lights that are left. */
    std::vector<uint32_t> tombstoneIndices; // places in the sources
    /* Where each light of the frame stands in the sources: false while the sources are the
       frame's own lights in the frame's own order, true while they are the lights the lists
       were composed in. The counters are indexed by the latter. */
    bool                  compositionOrder = false;
    std::vector<uint32_t> frameToSource;
    // Where each light of the composition stands in the frame, the map the slots that were
    // granted again on it are read through. Left over between frames only as capacity.
    std::vector<uint32_t> sourceToFrame;

    std::vector<int32_t> gridHead;
    std::vector<int32_t> gridNext;
    int   gridDims[3] = { 1, 1, 1 };
    float gridMins[3] = {};
    float gridCell = 1.0f;

    std::vector<uint32_t> offsets;    // numClusters + 1
    std::vector<uint64_t> list;
    uint32_t              listEntries = 0;
    /* Bumped by every change of the three above, so that the light manager can tell a frame
       whose lists are the ones it was last handed from a frame whose lists have to be built
       again: two publications of one generation carry the same words. */
    uint64_t              listGeneration = 0;

    std::vector<uint32_t> granted;
    std::vector<uint32_t> denied;

    RgClusterLightStats stats{};
    bool warnedAboutFullList = false;
    bool warnedAboutClusterClamp = false;

    // Sorted (uid, index) pairs of the live sources and of the frame being registered, used to
    // tell a light that moved from one that appeared or disappeared, and to put the per-light
    // counters of the composition back on the frame's own light order. The tombstones of the
    // sources are not in them: they are not lights of any frame.
    std::vector<std::pair<uint64_t, uint32_t>> prevUidIndex;
    std::vector<std::pair<uint64_t, uint32_t>> curUidIndex;
};

}
