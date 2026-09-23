// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/shadow_map.vert from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Shadow map vertex shader — depth-only pass rendering world geometry from
// the sun's point of view. No fragment shader.
// Ported from Quake 2 RTX (shadow_map.vert), GPL v2.
//


// HLSL counterpart of ShadowMap.vert. The stage is a vertex shader, so the profile is vs_6_2 and
// the entry point keeps the golden's name main.
//
// Spellings that had to change:
//   * gl_Position becomes the return value with an SV_Position semantic: an output of a vertex
//     shader needs a semantic in HLSL, and SV_Position is the builtin glslang also reports as
//     BuiltIn Position. The golden writes the builtin twice -- it assigns it and then negates its
//     y -- so the local float4 carries the same two steps, the negation included: the flip is the
//     golden's own line and not the one a GLSL front end applies for Vulkan, so dropping it would
//     change the image
//   * world_pos is an entry point parameter instead of a global input variable. HLSL requires a
//     semantic on every entry point parameter, and [[vk::location(0)]] pins the same location the
//     golden writes in layout(location = 0); the semantic index agrees with it
//   * push.view_projection_matrix * push.model_matrix * vec4(world_pos.xyz, 1.0) becomes
//     mul( mul( push.view_projection_matrix, push.model_matrix ), float4(world_pos.xyz, 1.0) ):
//     the operands keep the golden's order, which is what mul(m, v) means for the transposed
//     declaration, and the grouping stays left to right as the GLSL parses it. Both products are
//     measured on a scratch probe with literal matrices against the golden's own products
//   * layout(push_constant, std140) becomes a struct in a ConstantBuffer with
//     [[vk::push_constant]]. The two mat4 members land on offsets 0 and 64 on both halves, so no
//     [[vk::offset]] is needed, and each stays float4x4: a square shape is declared as it is
//
// One measured difference that is not a spelling choice: the two front ends declare the builtin
// differently. glslang puts Position on a member of the gl_PerVertex output block, dxc declares a
// plain output variable with the same BuiltIn and the same type -- both are valid Vulkan and the
// rasterizer receives the same builtin. The property checker reads the builtin decorations of the
// entry point interface, so it sees "builtin Position" on the HLSL side only, because the GLSL blob
// carries that decoration on the block member. It is reported in the wave-6 hand-off, not worked
// around here: the only HLSL way to hide it would be to declare and write the block's PointSize,
// ClipDistance and CullDistance outputs, which the golden never writes.
//
// What did not change: the parameter list of the golden, its whole body (one product chain and the
// y negation), the push constant member names view_projection_matrix and model_matrix, the name of
// the push constant instance push, and the vec4 construction float4(world_pos.xyz, 1.0) with the
// swizzle and the trailing component of the golden.

struct ShadowMapConstants
{
    float4x4 view_projection_matrix;
    float4x4 model_matrix;
};

[[vk::push_constant]] ConstantBuffer<ShadowMapConstants> push;

float4 main( [[vk::location(0)]] float4 world_pos : POSITION ) : SV_Position
{
    float4 position = mul( mul( push.view_projection_matrix, push.model_matrix ),
                           float4( world_pos.xyz, 1.0 ) );
    position.y = -position.y;
    return position;
}
