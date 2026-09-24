/*
* Copyright (c) 2026 Sultim Tsyrendashiev
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

// HLSL counterpart of GLSL/RhiDebugTrace.rmiss, the miss of the RHI debug pair: it writes the
// fixed background colour and touches no descriptor. RhiDebugPayload must stay identical in the
// pair's three files; the colour must stay the one RhiDebugTrace.rchit uses for a sky instance.
//
// The golden reads nothing and includes nothing - it declares the payload location as the literal
// 0 that PAYLOAD_INDEX_DEFAULT stands for - so the HLSL twin includes nothing either, and with no
// DESC_SET_* macro defined the pair's stages declare no resource but the raygen's.
//
// GLSL needs no attribute on main(); HLSL needs [shader("miss")], which makes the function the
// OpEntryPoint (MissKHR) of the lib_6_3 module. The payload is a module-scope
// rayPayloadInEXT global on the golden side and the entry's inout parameter on the HLSL side,
// as in RtMiss.rmiss.hlsl; it is the only payload type of the module, so dxc gives it location 0.

struct RhiDebugPayload
{
    float3 color;
};

[shader("miss")]
void main(inout RhiDebugPayload payload)
{
    // A near-black blue: the traced geometry is drawn unlit, so the background has to stay dark
    // for the hits to stand out against it.
    payload.color = float3(0.01, 0.01, 0.02);
}
