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

// HLSL counterpart of GLSL/RtClsOpaque.rchit, the fully opaque closest hit of the base: it copies
// the two barycentric coordinates of the hit and packs the instance and the triangle it belongs to
// into the default payload, which the raygen side then decodes.
//
// GLSL needs no attribute on main(); HLSL needs [shader("closesthit")] for the OpEntryPoint
// (ClosestHitKHR), and that entry is what makes the payload and the hit attributes exist.
//
// Spellings that had to change:
//   * the golden's `layout(location = PAYLOAD_INDEX_DEFAULT) rayPayloadInEXT ShPayload g_payload`
//     global becomes the entry's first `inout` parameter, keeping the golden's name and type. As in
//     the two misses there is no location to copy: spirv-dis shows both compilers emitting
//     `%g_payload = OpVariable ... IncomingRayPayloadKHR` with NO Location decoration, the
//     golden's layout qualifier staying in the source. `[[vk::location(...)]]` on the parameter is
//     accepted and silently dropped by dxc, so it is not written.
//   * the golden's `hitAttributeEXT vec2 inBaryCoords` becomes the entry's second parameter, the
//     hit attributes. dxc demands a user defined struct there ("attributes parameter must be a
//     user-defined type composed of only numeric types": a bare float2 and a float2 with
//     SV_Barycentrics are both rejected), so the struct keeps the golden's member name and the
//     member is what the body reads. Measured: the HitAttributeKHR variable then holds the same
//     two floats at offset 0 as the golden's vec2 variable, and neither compiler decorates it with
//     a Location, so the pair describes the attributes identically.
//   * gl_InstanceID -> InstanceIndex() and gl_InstanceCustomIndexEXT -> InstanceID(), NOT the
//     other way round. dxc's two intrinsics are named after the DirectX meaning: InstanceID() is
//     the user-provided instance identifier, which is SPIR-V's InstanceCustomIndexKHR, and
//     InstanceIndex() is the index inside the acceleration structure, which is SPIR-V's InstanceId,
//     the builtin behind GLSL's gl_InstanceID. Measured with a one-intrinsic scratch pair
//     (%TEMP%\fleet_w7e\dev\probe13_instanceid.hlsl, probe14_instanceindex.hlsl): InstanceID()
//     emits BuiltIn InstanceCustomIndexKHR and InstanceIndex() emits BuiltIn InstanceId. The names are the only tell in the source, so the
//     CheckShaderProperties.py property set cannot see a swap -- it compares the set of builtins,
//     not which one reaches which argument.
//   * the two index packers are the ported header functions of ShaderCommonHLSLFunc.hlsli, called
//     with the same names and the same argument order as in the golden. Their arguments are the
//     uint intrinsics, truncated by the explicit `(int)` cast: the golden's builtins are signed
//     ints and the packers take ints, dxc's intrinsics are uint. The cast keeps the bits (an
//     instance id below 256 and a 24 bit custom index, as the golden's comment states), and it
//     keeps the module warning free.
//
// What did not change: the order of the three payload writes, the packers themselves, the include
// (ShaderCommonGLSLFunc.h -> ShaderCommonHLSLFunc.hlsli) and the absence of any descriptor of its
// own. There is no arithmetic of this file beyond the two packer calls; the packed values are
// measured in the report.

#include "ShaderCommonHLSLFunc.hlsli"

struct HitAttributes
{
    float2 inBaryCoords;
};

[shader("closesthit")]
void main(inout ShPayload g_payload, in HitAttributes attribs)
{
    g_payload.baryCoords = attribs.inBaryCoords;
    g_payload.instIdAndIndex = packInstanceIdAndCustomIndex((int)InstanceIndex(), (int)InstanceID());
    g_payload.geomAndPrimIndex = packGeometryAndPrimitiveIndex((int)GeometryIndex(), (int)PrimitiveIndex());
}
