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

#include "WorldLights.h"

#include <cstdio>

#include "RgException.h"
#include "UserFunction.h"

namespace vkpt
{

template <typename T>
static void CopyTable(const T *pData, uint32_t count, std::vector<T> &dst, const char *pWhatIsMissing)
{
    if (count == 0)
    {
        dst.clear();
        return;
    }

    if (pData == nullptr)
    {
        throw RgException(RG_WRONG_ARGUMENT, pWhatIsMissing);
    }

    dst.assign(pData, pData + count);
}

void WorldLights::Upload(const RgWorldLightsUploadInfo &uploadInfo, UserPrint *pUserPrint)
{
    CopyTable(uploadInfo.pClusterMins, uploadInfo.numClusters, clusterMins, "World lights: no cluster bounds");
    CopyTable(uploadInfo.pClusterMaxs, uploadInfo.numClusters, clusterMaxs, "World lights: no cluster bounds");
    CopyTable(uploadInfo.pClusterFlags, uploadInfo.numClusters, clusterFlags, "World lights: no cluster flags");
    CopyTable(uploadInfo.pVisOffsets, uploadInfo.numClusters, visOffsets, "World lights: no PVS offsets");
    CopyTable(uploadInfo.pFaces, uploadInfo.numFaces, faces, "World lights: no face array");
    CopyTable(uploadInfo.pFaceVertices, uploadInfo.numFaceVertices, faceVertices, "World lights: no face vertex array");
    CopyTable(uploadInfo.pVisData, uploadInfo.visDataSize, visData, "World lights: no PVS data");

    pvsRowBytes = uploadInfo.pvsRowBytes;

    /* The sky visibility is sized by the cluster count, so it is copied by hand rather than
       through CopyTable, and a missing table is not an error: the light manager keeps every
       cluster tracing then. */
    clusterSkyVisibility.clear();
    if (uploadInfo.pClusterSkyVisibility != nullptr && uploadInfo.numClusters > 0)
    {
        clusterSkyVisibility.assign(uploadInfo.pClusterSkyVisibility,
                                    uploadInfo.pClusterSkyVisibility + (uploadInfo.numClusters + 7) / 8);
    }

    if (uploadInfo.flags & RG_WORLD_LIGHTS_UPLOAD_PRINT_STATS_BIT)
    {
        Print(pUserPrint);
    }
}

const uint8_t *WorldLights::GetClusterVis(uint32_t cluster) const
{
    if (cluster >= visOffsets.size() || visOffsets[cluster] < 0)
    {
        return nullptr;
    }

    const uint32_t offset = uint32_t(visOffsets[cluster]);

    if (offset + pvsRowBytes > visData.size())
    {
        return nullptr;
    }

    return visData.data() + offset;
}

size_t WorldLights::GetMemoryBytes() const
{
    return clusterMins.size() * sizeof(RgFloat3D) + clusterMaxs.size() * sizeof(RgFloat3D) +
           clusterFlags.size() +
           faces.size() * sizeof(RgWorldLightFace) + faceVertices.size() * sizeof(RgVertex) +
           visData.size() + visOffsets.size() * sizeof(int32_t);
}

void WorldLights::Print(const UserPrint *pUserPrint) const
{
    uint32_t masked = 0, inlineModels = 0, dynamic = 0, clustersWithoutPvs = 0;

    for (const RgWorldLightFace &f : faces)
    {
        masked       += (f.flags & RG_WORLD_LIGHT_FACE_MASKED_BIT) ? 1 : 0;
        inlineModels += (f.flags & RG_WORLD_LIGHT_FACE_INLINE_MODEL_BIT) ? 1 : 0;
        dynamic      += f.isStatic ? 0 : 1;
    }

    for (uint32_t i = 0; i < GetClusterCount(); i++)
    {
        clustersWithoutPvs += (GetClusterVis(i) == nullptr) ? 1 : 0;
    }

    char buffer[512];

    snprintf(buffer, sizeof(buffer),
             "world lights: %u clusters (%u without a PVS row), %u bytes per PVS row, visdata %u bytes (%.1f MB)\n",
             GetClusterCount(), clustersWithoutPvs, pvsRowBytes, GetVisDataSize(), float(GetVisDataSize()) / 1048576.0f);
    pUserPrint->Print(buffer);

    snprintf(buffer, sizeof(buffer),
             "world lights: %u emissive faces (%u masked, %u of inline models, %u not static) with %u corners\n",
             GetFaceCount(), masked, inlineModels, dynamic, GetFaceVertexCount());
    pUserPrint->Print(buffer);

    snprintf(buffer, sizeof(buffer),
             "world lights: tables take %.1f KB\n", float(GetMemoryBytes()) / 1024.0f);
    pUserPrint->Print(buffer);
}

}
