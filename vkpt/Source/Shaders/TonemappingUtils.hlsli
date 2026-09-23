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
// Must be included AFTER ShaderCommonHLSLFunc.hlsli (needs COMPUTE_LUM_HISTOGRAM_BIN_COUNT).
//
// HLSL counterpart of TonemappingUtils.glsl. Like the GLSL one it declares no resource of its
// own: the consumers of the golden pull in ShaderCommonGLSLFunc.h first, and that is where
// COMPUTE_LUM_HISTOGRAM_BIN_COUNT and -- through the generated header -- the framebuffer set
// live. The port declares nothing either, so it gets no probe of its own and is pinned by the
// pair of its first consumer, GLSL/CmLuminanceHistogram.comp <-> CmLuminanceHistogram.comp.hlsl,
// which includes this file.
//
// Spellings that had to change:
//   * the #ifndef TONEMAPPING_UTILS_GLSL_ / #define TONEMAPPING_UTILS_GLSL_ guard ->
//     TONEMAPPING_UTILS_HLSLI_, the guard spelling every ported header of the base uses
//   * const float x = ... at file scope -> static const float x = ...: a global without static
//     is an *external* global in HLSL and dxc places it in the $Globals constant buffer with the
//     initializer ignored (measured: "variable 'min_log_luminance' will be placed in $Globals so
//     initializer ignored"), so the constant would become a cbuffer read. static makes it the
//     compile time constant the golden has, the same spelling EfChromaticAberration.comp.hlsl
//     and EfCrtDecode.comp.hlsl use for their file scope constants
//   * the include-order note above names ShaderCommonGLSLFunc.h in the golden and
//     ShaderCommonHLSLFunc.hlsli here: both headers are emitted from the same walk and carry the
//     same macro, only the file name differs
//
// What did not change: every macro (HISTOGRAM_BINS, FIXED_POINT_FRAC_BITS,
// FIXED_POINT_FRAC_MULTIPLIER), every constant name, and the arithmetic of the constants.
// log_luminance_scale and log_luminance_bias are the golden's expressions with the golden's
// operand order, and HISTOGRAM_BINS is still COMPUTE_LUM_HISTOGRAM_BIN_COUNT. The file is
// nothing but constants, so every site of it folds to a literal, measured on both halves:
// min_log_luminance -24, max_log_luminance 8, max_log_luminance - min_log_luminance 32,
// log_luminance_scale 0.03125, log_luminance_bias 0.75, exp2(min_log_luminance) 5.96046448e-08
// and float(FIXED_POINT_FRAC_MULTIPLIER) 128. The file holds no function, no matrix, no resource
// access and no control flow, so none of the matrix rules of ShaderCommonHLSL.hlsli applies to it.

#ifndef TONEMAPPING_UTILS_HLSLI_
#define TONEMAPPING_UTILS_HLSLI_

#define HISTOGRAM_BINS COMPUTE_LUM_HISTOGRAM_BIN_COUNT

// Conversion from floating-point to fixed-point for atomic histogram adds.
#define FIXED_POINT_FRAC_BITS 7
#define FIXED_POINT_FRAC_MULTIPLIER (1 << FIXED_POINT_FRAC_BITS)

// Maximum log (base 2) luminance values (photographic stops) represented in
// the tone mapping histogram. Anything below min_log_luminance is not
// counted and becomes inky black on screen.
static const float min_log_luminance = -24.0;
static const float max_log_luminance = 8.0;

// Map [min_log_luminance, max_log_luminance] to [0, 1] with a single FMA:
//   x/(max-min) - min/(max-min)
static const float log_luminance_scale = 1.0 / (max_log_luminance - min_log_luminance);
static const float log_luminance_bias = -min_log_luminance * log_luminance_scale;

#endif // TONEMAPPING_UTILS_HLSLI_
