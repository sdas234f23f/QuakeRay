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

#version 460

layout(location = 0) in vec4 world_pos;

layout(push_constant, std140) uniform ShadowMapConstants
{
    mat4 view_projection_matrix;
} push;

void main()
{
    gl_Position = push.view_projection_matrix * vec4(world_pos.xyz, 1.0);
    gl_Position.y = -gl_Position.y;
}
