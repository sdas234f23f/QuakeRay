// Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Copyright (c) 2026 QuakeRay contributors
//
// This file is a port of shader/tone_mapping_utils.glsl from Quake 2 RTX (https://github.com/NVIDIA/Q2RTX),
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
// Shared constants for the Q2RTX-style noise-aware tone mapper.
// Ported from Quake 2 RTX (tone_mapping_utils.glsl), GPL v2.
// Must be included AFTER ShaderCommonGLSLFunc.h (needs COMPUTE_LUM_HISTOGRAM_BIN_COUNT).
//

#ifndef TONEMAPPING_UTILS_GLSL_
#define TONEMAPPING_UTILS_GLSL_

#define HISTOGRAM_BINS COMPUTE_LUM_HISTOGRAM_BIN_COUNT

// Conversion from floating-point to fixed-point for atomic histogram adds.
#define FIXED_POINT_FRAC_BITS 7
#define FIXED_POINT_FRAC_MULTIPLIER (1 << FIXED_POINT_FRAC_BITS)

// Maximum log (base 2) luminance values (photographic stops) represented in
// the tone mapping histogram. Anything below min_log_luminance is not
// counted and becomes inky black on screen.
const float min_log_luminance = -24.0;
const float max_log_luminance = 8.0;

// Map [min_log_luminance, max_log_luminance] to [0, 1] with a single FMA:
//   x/(max-min) - min/(max-min)
const float log_luminance_scale = 1.0 / (max_log_luminance - min_log_luminance);
const float log_luminance_bias = -min_log_luminance * log_luminance_scale;

#endif // TONEMAPPING_UTILS_GLSL_
