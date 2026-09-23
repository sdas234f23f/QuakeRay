// Copyright (c) 2022 Sultim Tsyrendashiev
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// HLSL counterpart of RtRaygenPrimary.rgen. The file is the golden's three lines: the
// specialization constant, MATERIAL_MAX_ALBEDO_LAYERS and the entry point macro.
//
//   * layout (constant_id = 0) const uint maxAlbedoLayerCount = 0 -> [[vk::constant_id(0)]]
//     const uint maxAlbedoLayerCount = 0: the same SpecId 0, uint, default 0.
//   * #include "RaygenPrimary.inl" -> #include "RaygenPrimary.hlsli".
//
// No main() is defined here. The header's main(), guarded by RAYGEN_PRIMARY_SHADER and carrying
// RAYGEN_PRIMARY_ENTRY_ATTR, is the raygen entry point, exactly as the golden's guarded main() is.

[[vk::constant_id(0)]] const uint maxAlbedoLayerCount = 0;
#define MATERIAL_MAX_ALBEDO_LAYERS maxAlbedoLayerCount

#define RAYGEN_PRIMARY_SHADER
#include "RaygenPrimary.hlsli"
