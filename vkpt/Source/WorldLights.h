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

#include <cstddef>
#include <vector>

#include <vkpt/vkpt.h>

#include "Common.h"

namespace vkpt
{

class UserPrint;

// Owns the world tables of a map: the clusters (BSP leafs), the compressed PVS and the
// emissive faces with their corners. The host builds them once per map load and stops
// touching them; the light polygons are collected from these tables by this module
// (the Q2RTX bsp_mesh.c collection lands here).
class WorldLights
{
public:
    WorldLights() = default;
    ~WorldLights() = default;

    WorldLights(const WorldLights &other) = delete;
    WorldLights(WorldLights &&other) noexcept = delete;
    WorldLights &operator=(const WorldLights &other) = delete;
    WorldLights &operator=(WorldLights &&other) noexcept = delete;

    void Upload(const RgWorldLightsUploadInfo &uploadInfo, UserPrint *pUserPrint);

    uint32_t GetClusterCount() const { return uint32_t(clusterMins.size()); }
    uint32_t GetFaceCount() const { return uint32_t(faces.size()); }
    uint32_t GetFaceVertexCount() const { return uint32_t(faceVertices.size()); }
    uint32_t GetVisDataSize() const { return uint32_t(visData.size()); }
    uint32_t GetPvsRowBytes() const { return pvsRowBytes; }

    const RgFloat3D        &GetClusterMin(uint32_t index) const { return clusterMins[index]; }
    const RgFloat3D        &GetClusterMax(uint32_t index) const { return clusterMaxs[index]; }
    const RgWorldLightFace &GetFace(uint32_t index) const { return faces[index]; }
    const RgVertex         &GetFaceVertex(uint32_t index) const { return faceVertices[index]; }

    // A solid leaf holds no light the camera could see, so the list builder skips it.
    bool IsClusterSolid(uint32_t cluster) const
    {
        return cluster < clusterFlags.size() && (clusterFlags[cluster] & RG_WORLD_CLUSTER_SOLID_BIT) != 0;
    }

    // Row of the compressed PVS of a cluster, or nullptr when it sees everything.
    const uint8_t *GetClusterVis(uint32_t cluster) const;

    size_t GetMemoryBytes() const;

private:
    void Print(const UserPrint *pUserPrint) const;

private:
    std::vector<RgFloat3D>        clusterMins;
    std::vector<RgFloat3D>        clusterMaxs;
    std::vector<uint8_t>          clusterFlags;
    std::vector<RgWorldLightFace> faces;
    std::vector<RgVertex>         faceVertices;
    std::vector<uint8_t>          visData;
    std::vector<int32_t>          visOffsets;
    uint32_t                      pvsRowBytes = 0;
};

}
