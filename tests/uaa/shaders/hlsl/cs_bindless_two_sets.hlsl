/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

#if defined(UAA_EXPLICIT_SETS)
[[vk::binding(0, 1)]] Texture2D<float4> First[] : register(t0, space0);
[[vk::binding(0, 0)]] SamplerState Samp : register(s0, space2);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Out : register(u1, space2);
#else
[[vk::binding(0, 0)]] Texture2D<float4> First[] : register(t0, space0);
[[vk::binding(0, 1)]] SamplerState Samp : register(s0, space2);
[[vk::binding(1, 1)]] RWStructuredBuffer<uint> Out : register(u1, space2);
#endif
[[vk::binding(0, 2)]] Texture2D<float4> Second[] : register(t0, space1);

uint packColour(float4 colour)
{
    uint4 c = uint4(colour * 255.0f + 0.5f);
    return c.x | (c.y << 8) | (c.z << 16) | (c.w << 24);
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float2 uv = float2(0.25f, 0.25f);
    Out[0] = packColour(First[NonUniformResourceIndex(id.x + 1)].SampleLevel(Samp, uv, 0));
    Out[1] = packColour(Second[NonUniformResourceIndex(id.x + 6)].SampleLevel(Samp, uv, 0));
}
