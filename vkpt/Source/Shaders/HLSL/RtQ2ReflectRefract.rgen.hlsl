// Copyright (C) 2018 Christoph Schied
// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/reflect_refract.rgen from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
// which is distributed under the terms of the GNU General Public License
// version 2.  It has been adapted to the renderer interface of this project.
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
// Q2RTX-style reflection/refraction raygen (used on the Q2 core path).
// Ported from Q2RTX reflect_refract.rgen; see the Q2_REFL_REFR_SHADER section
// in RaygenPrimary.inl for the implementation.
//

// HLSL counterpart of RtQ2ReflectRefract.rgen. The file is the golden's three lines: the
// specialization constant, MATERIAL_MAX_ALBEDO_LAYERS and the entry point macro.
//
//   * layout (constant_id = 0) const uint maxAlbedoLayerCount = 0 -> [[vk::constant_id(0)]]
//     const uint maxAlbedoLayerCount = 0: the same SpecId 0, uint, default 0.
//   * #include "RaygenPrimary.inl" -> #include "RaygenPrimary.hlsli".
//
// No main() is defined here. The header's main(), guarded by Q2_REFL_REFR_SHADER and carrying
// RAYGEN_PRIMARY_ENTRY_ATTR, is the raygen entry point, exactly as the golden's guarded main() is.

[[vk::constant_id(0)]] const uint maxAlbedoLayerCount = 0;
#define MATERIAL_MAX_ALBEDO_LAYERS maxAlbedoLayerCount

#define Q2_REFL_REFR_SHADER
#include "RaygenPrimary.hlsli"
