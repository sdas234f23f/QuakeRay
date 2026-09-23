// Copyright (c) 2021 Sultim Tsyrendashiev
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

// HLSL counterpart of GLSL/RtAlphaTest.rahit, the alpha-test any-hit: it samples the albedo of the
// first material at the barycentric coordinate of the intersection and ignores the intersection
// when the combined alpha falls under the threshold, so that the ray continues to the next
// candidate. The turb-warp UV of TurbWarp.h is applied here because the closest hit applies it too,
// so the alpha test samples the very texel the opaque path would have.
//
// GLSL needs no attribute on main(); HLSL needs [shader("anyhit")] for the OpEntryPoint
// (AnyHitKHR).
//
// Spellings that had to change:
//   * the hit attributes: the golden's `hitAttributeEXT vec2 inBaryCoords` becomes the entry's
//     second parameter, which dxc demands to be a user defined struct (a bare float2 is rejected),
//     so the struct keeps the golden's name as its member and the body reads `attribs.inBaryCoords`
//     where the golden read `inBaryCoords`. The variable is the same two floats at offset 0 and
//     neither compiler decorates it with a Location.
//   * the payload: the golden's any-hit declares none, and dxc does not accept that -- measured:
//     "incorrect number of entry parameters for raytracing stage 'anyhit': 1 parameter(s)
//     provided, expected two parameters for payload and attributes", and an `in` payload is
//     refused too ("payload parameter 'g_payload' must be 'inout'"). The port therefore declares
//     an `inout` payload and never reads or writes a field of it. Measured on the shipped pair:
//     dxc emits a round trip of the whole payload for every any-hit (`OpLoad` then two `OpStore`s
//     of the very same value, one per exit, with and without IgnoreHit), which restores the
//     payload bit for bit -- the value never changes, so the golden's behaviour is kept. The
//     declared type is the 4 byte ShPayloadShadow, not the 16 byte ShPayload: the round trip is
//     an access to the payload the caller of the trace passes, so the 4 byte view stays inside
//     the payload of a shadow ray as well, while a 16 byte view would reach 12 bytes past it. The
//     name is the golden's name for the shadow payload (RaygenCommon.h).
//     Neither half decorates the variable with a Location: measured with spirv-dis, glslang emits
//     no Location for an IncomingRayPayloadKHR variable, and dxc accepts `[[vk::location(1)]]` on
//     the parameter and drops it silently (scratch probe).
//   * gl_InstanceID -> InstanceIndex() and gl_InstanceCustomIndexEXT -> InstanceID(), NOT the other
//     way round: in dxc the user-provided instance id is the custom index of SPIR-V, and
//     InstanceIndex() is SPIR-V's InstanceId, the builtin behind gl_InstanceID (measured with a
//     one-intrinsic scratch pair, %TEMP%\fleet_w7e\dev\probe13_instanceid.hlsl and
//     probe14_instanceindex.hlsl; the checker compares the set of
//     builtins only, so a swap is invisible to it). gl_GeometryIndexEXT -> GeometryIndex() and
//     gl_PrimitiveID -> PrimitiveIndex() keep their names. The `(int)` casts truncate the uint
//     intrinsics to the ints of getTriangle(), exactly as glslang's builtins are ints.
//   * vec3(...) -> float3(...), the one constructor of the golden, with every argument unchanged.
//   * `tr.layerTexCoord[0] * baryCoords` -> `mul(tr.layerTexCoord[0], baryCoords)`: the member is
//     the golden's mat3x2 declared as a float2x3 (Structs.hlsli), and the product rule of
//     ShaderCommonHLSL.hlsli says to keep the golden's operand order, which makes this a mat3x2 *
//     vec3 -> vec2 on both sides. The mirrored `mul(baryCoords, tr.layerTexCoord[0])` is an
//     implicit truncation in dxc (two -Wconversion warnings) and folds to another number, and the
//     GLSL `baryCoords * tr.layerTexCoord[0]` does not compile at all -- measured in
//     %TEMP%\fleet_w7e\fold\w7e_fold_mirror.*, where the correct spelling folds to
//     (1.625, -0.1875) and the control to (0.4375, 0.625).
//   * `getTextureSample(t, texCoord)` -> `getTextureSampleLod(t, texCoord, 0.0)`: dxc rejects an
//     implicit lod sample outside fragment and compute shaders ("sampling with implicit lod is only
//     allowed in fragment and compute shaders"). The golden's implicit lod is the base level in a
//     non-fragment stage, and glslang itself compiles this very call into
//     `OpImageSampleExplicitLod ... Lod %float_0`, the instruction the explicit 0.0 emits, so the
//     two halves hold the same sample instruction.
//   * ignoreIntersectionEXT -> IgnoreHit(), the golden's meaning of abandoning this intersection.
//
// What did not change: MATERIAL_MAX_ALBEDO_LAYERS 0 and its comment, the three DESC_SET_* indices,
// the include order (ShaderCommonGLSLFunc.h then TurbWarp.h -> ShaderCommonHLSLFunc.hlsli then
// TurbWarp.hlsli), ALPHA_THRESHOLD 0.5, the whole arithmetic of the barycentric coordinates and of
// the alpha test, the index `tr.materials[0][MATERIAL_ALBEDO_ALPHA_INDEX]` and the component wise
// `* tr.materialColors[0]`, and the turb-warp call itself.

// don't need "processAlbedo" function
#define MATERIAL_MAX_ALBEDO_LAYERS 0

#define DESC_SET_GLOBAL_UNIFORM 2
#define DESC_SET_VERTEX_DATA 3
#define DESC_SET_TEXTURES 4
#include "ShaderCommonHLSLFunc.hlsli"

// the alpha test must sample the same texel as the closest hit shader
#include "TurbWarp.hlsli"

struct HitAttributes
{
    float2 inBaryCoords;
};


#define ALPHA_THRESHOLD 0.5


[shader("anyhit")]
void main(inout ShPayloadShadow g_payloadShadow, in HitAttributes attribs)
{
	const ShTriangle tr = getTriangle((int)InstanceIndex(), (int)InstanceID(), (int)GeometryIndex(), (int)PrimitiveIndex());

	const float3 baryCoords = float3(1.0f - attribs.inBaryCoords.x - attribs.inBaryCoords.y, attribs.inBaryCoords.x, attribs.inBaryCoords.y);
    const float2 texCoord = getSurfaceTexCoord(tr.geometryInstanceFlags, mul(tr.layerTexCoord[0], baryCoords));
 
	// check only first material's albedo
 	const float4 color = getTextureSampleLod(tr.materials[0][MATERIAL_ALBEDO_ALPHA_INDEX], texCoord, 0.0) * tr.materialColors[0];

	// if must be discarded
	if ((color.r + color.g + color.b) / 3 * color.a + color.a < ALPHA_THRESHOLD)
	{
		// then ignore this intersection, so it won't be the closest hit
		IgnoreHit();
	}

	// otherwise, do nothing, as this intersection 
	// can or can't be the closest hit
}
