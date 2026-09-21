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
// Rules to keep the port honest, apply them everywhere in the HLSL shader base:
//   * matrices: declare the shape transposed, GLSL `matCxR` becomes HLSL `floatRxC` (a
//     square `matN` becomes `floatNxN`). An element access is then copied as is, `m[i][j]`
//     stays `m[i][j]`, and it is legal exactly where the GLSL one was. Only a product
//     changes: GLSL `m * v` becomes HLSL `mul(v, m)`, GLSL `a * b` becomes `mul(b, a)`, and
//     `transpose(...)` is copied as is. For the very same bytes dxc declares such a matrix
//     as RowMajor where glslang declares it as ColMajor, i.e. the two languages read the
//     same bytes as transposed matrices, and that is exactly what the transposed
//     declaration and the product rule compensate for. With this dxc lands every offset,
//     MatrixStride and ArrayStride on the same byte as glslang, so matrices never need
//     [[vk::offset]].
//   * struct members: dxc packs nested struct members on 4-byte boundaries, GLSL std430
//     uses the alignment of the member type, so e.g. a float2 after a float lands on
//     offset 4 instead of 8. Where the checker reports such a difference, pin the member
//     with [[vk::offset(N)]].


// Utils.hlsli comes first, exactly as Utils.h does before the generated header on the GLSL side:
// the generated header implements the shared exponent pack and unpack of its framebuffers with
// encodeE5B9G9R9 and decodeE5B9G9R9, which Utils.hlsli has to declare.
#include "Utils.hlsli"
#include "../Generated/ShaderCommonHLSL.hlsli"


#ifdef DESC_SET_GLOBAL_UNIFORM

[[vk::binding(BINDING_GLOBAL_UNIFORM, DESC_SET_GLOBAL_UNIFORM)]] ConstantBuffer<ShGlobalUniform> globalUniform;

#endif // DESC_SET_GLOBAL_UNIFORM

