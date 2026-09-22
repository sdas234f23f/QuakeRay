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


// HLSL counterpart of the generated GLSL header. The constants, the structs and the whole
// framebuffer set come from Generated/ShaderCommonHLSL.hlsli, which GenerateShaderCommon.py emits
// from the very same walk that produces Generated/ShaderCommonGLSL.h, so the two languages can
// not drift apart. What the generator does not cover is added here.
//
// The generated header takes the set index of the framebuffers from DESC_SET_FRAMEBUFFERS, so it
// has to be included after that macro is defined. The path is spelled out relative to this file
// because a header of the same name exists in both folders.
//
// Rules to keep the port honest, apply them everywhere in the HLSL shader base. The matrix rules
// are the parts of one fact, and that fact is measurable instead of assumed: for the same bytes
// dxc and glslang declare transposed storage types (GLSL `matCxR` against HLSL `floatRxC`,
// ColMajor against RowMajor) and land every Offset, MatrixStride and ArrayStride on the same byte,
// which the AccessChains of the probe pair Structs.probe.comp confirm one to one, off-diagonal
// accesses included. With the storage type transposed on one side and the majorness decorators
// swapping the roles of the two indices, both languages show the shader the very same logical
// matrix, so every product and every index written on it keeps the meaning it had in the GLSL.
//
//   * matrices, declaration: declare the shape transposed, GLSL `matCxR` becomes HLSL `floatRxC`
//     (a square `matN` becomes `floatNxN`). For the very same bytes dxc then declares such a
//     matrix as RowMajor where glslang declares it as ColMajor, and dxc lands every Offset,
//     MatrixStride and ArrayStride on the same byte as glslang, so a matrix never needs
//     [[vk::offset]].
//   * matrices, product: keep the order of the operands, GLSL `m * v` becomes HLSL `mul(m, v)` and
//     GLSL `a * b` becomes `mul(a, b)`. The mirrored spelling `mul(v, m)` multiplies by the
//     transpose instead. On a square matrix that mistake is invisible in the image, because
//     `mul(v, M)` equals `mul(transpose(M), v)`, which is why it survives review for a long time;
//     on a non-square matrix it is a real defect, and dxc even reports it as an implicit
//     truncation on a `float2x3`. Neither this rule nor the constructor rule below has to be
//     argued: with literal operands both compilers fold the product into a constant, and the
//     golden spelling and the HLSL one fold to the same numbers while the mirrored spelling and
//     the untransposed argument list fold to different ones (step 5 and step 6 of the measurement
//     in §14.5 of the plan).
//   * matrices, element access: a double index swaps, GLSL `m[i][j]` becomes HLSL `m[j][i]`. On a
//     non-square matrix that spelling leaves the row range, so `transpose(m)[i][j]` is the form to
//     prefer: it is correct for every shape and needs no local. A single index is a column in GLSL
//     and a row in HLSL, so GLSL `m[i]` becomes `getColumn(m, i)` of ShaderCommonHLSLFunc.hlsli,
//     the row of the transposed matrix, which stays in range for a non-square matrix as well. A
//     bare `transpose(...)` without an index on it is copied as is. A single index on the left of
//     an assignment has no counterpart at all, because `m[i] = v` writes a row, so a golden member
//     that is filled column by column is built as rows and transposed instead,
//     `tr.positions = transpose(float3x3(a.position.xyz, b.position.xyz, c.position.xyz));` —
//     steps 7 and 8 of the same measurement.
//   * matrices, local constructors: HLSL fills rows where GLSL fills columns, for the scalar and
//     the vector argument list alike, so the argument list of GLSL is transcribed transposed
//     (GLSL `mat3x2(c0, c1, c2)` with `ck` a `vec2` becomes
//     `float2x3(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y)`).
//   * matrices in a buffer, in a descriptor or in the push constants: the declaration rule above
//     applies to them exactly as it does to a local, so their shape is declared transposed too and
//     they land on the same bytes on both sides, which is why such a member needs no [[vk::offset]].
//     The product rule and the element access rule follow the languages rather than the storage, so
//     they stay binding for these matrices as well.
//   * struct members: dxc packs nested struct members on 4-byte boundaries, GLSL std430
//     uses the alignment of the member type, so e.g. a float2 after a float lands on
//     offset 4 instead of 8. Where the checker reports such a difference, pin the member
//     with [[vk::offset(N)]].
//   * include guards: keep the mechanism of the golden and guard with `#ifndef STEM_HLSLI_` rather
//     than `#pragma once`, and keep a golden without a guard unguarded. `#pragma once` is keyed to
//     the spelling of a path: one file reached as "Structs.hlsli" and as "./Structs.hlsli" in the
//     same translation unit is processed twice and redefines every struct (measured: two includes
//     of one guarded file under the same spelling compile, under a second spelling dxc reports
//     "redefinition"). The goldens that are included more than once on purpose must therefore stay
//     unguarded: RaygenCommon.h includes HitInfo.inl three times behind HITINFO_INL_PRIM, _RFL and
//     _INDIR, CmVertexPreprocess.comp includes VertexPreprocessPartial.inl three times behind
//     VERTEX_PREPROCESS_PARTIAL_*, and EfSimple.inl includes EfCommon.inl, so HitInfo.hlsli,
//     VertexPreprocessPartial.hlsli, EfSimple.hlsli and EfCommon.hlsli carry no guard either. Every
//     other header of the base is guarded, whatever the golden used, because a diamond include of a
//     ported header is not an error to be worked around but a guard to be written.


// Utils.hlsli comes first, exactly as Utils.h does before the generated header on the GLSL side:
// the generated header implements the shared exponent pack and unpack of its framebuffers with
// encodeE5B9G9R9 and decodeE5B9G9R9, which Utils.hlsli has to declare.

#ifndef SHADER_COMMON_HLSL_HLSLI_
#define SHADER_COMMON_HLSL_HLSLI_
#include "Utils.hlsli"
#include "../Generated/ShaderCommonHLSL.hlsli"


#ifdef DESC_SET_GLOBAL_UNIFORM

[[vk::binding(BINDING_GLOBAL_UNIFORM, DESC_SET_GLOBAL_UNIFORM)]] ConstantBuffer<ShGlobalUniform> globalUniform;

#endif // DESC_SET_GLOBAL_UNIFORM

#endif // SHADER_COMMON_HLSL_HLSLI_
