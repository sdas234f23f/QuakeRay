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

// HLSL counterpart of GLSL/RtMiss.rmiss, the default (sky) miss: it writes the default payload,
// and the payload helper of RaygenCommon.h leaves it zeroed, so the body stays empty.
//
// GLSL needs no attribute on main(); HLSL needs [shader("miss")], which is what makes the function
// the OpEntryPoint (MissKHR) of the lib_6_3 module. The entry stays named main, so the blob name
// and the host are untouched.
//
// The payload is the one spelling that had to change:
//   * GLSL declares the golden's `layout(location = PAYLOAD_INDEX_DEFAULT) rayPayloadInEXT ShPayload
//     g_payload` as a module scope global. HLSL has no ray payload global in a miss shader: dxc
//     represents it as the entry's `inout` parameter, so the parameter carries the golden's name
//     and type. The location itself has no spelling to copy: measured with spirv-dis, glslang
//     emits the variable as `%g_payload = OpVariable ... IncomingRayPayloadKHR` with NO Location
//     decoration (the golden's layout qualifier stays in the source), and dxc emits the very same
//     variable without a decoration as well, so both halves describe the payload identically.
//     An explicit `[[vk::location(0)]]` on the parameter was measured to be accepted and silently
//     dropped by dxc (scratch probe), so it is not written.
// The body is empty on purpose: dxc keeps an unused `inout` payload parameter in the entry point
// interface (measured: the empty main() still declares the IncomingRayPayloadKHR variable), which
// is exactly what the golden's empty main() leaves behind.
//
// Nothing else is spelled: the includes match the golden's (ShaderCommonGLSLFunc.h ->
// ShaderCommonHLSLFunc.hlsli), no DESC_SET_* macro is needed to reach the payload, and there is no
// arithmetic at all in this stage.

#include "ShaderCommonHLSLFunc.hlsli"

[shader("miss")]
void main(inout ShPayload g_payload)
{
}
