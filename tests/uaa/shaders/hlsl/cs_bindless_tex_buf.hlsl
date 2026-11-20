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

// Exercises bindless texture and structured buffer arrays in descriptor set 0
// paired with regular index/output bindings in descriptor set 1.

[[vk::binding(0, 0)]] Texture2D<float4> TexTable[] : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<uint> BufTable[] : register(t0, space1);

[[vk::binding(0, 1)]] StructuredBuffer<uint> TexIndices : register(t0, space2);
[[vk::binding(1, 1)]] StructuredBuffer<uint> BufIndices : register(t1, space2);
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> Out : register(u2, space2);
[[vk::binding(3, 1)]] SamplerState Samp : register(s3, space2);
[[vk::binding(4, 1)]] cbuffer Params : register(b4, space2)
{
    uint count;
    uint mode;
    uint p0;
    uint p1;
};

uint packRGBA8(float4 c)
{
    uint r = uint(clamp(c.r, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint g = uint(clamp(c.g, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint b = uint(clamp(c.b, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint a = uint(clamp(c.a, 0.0f, 1.0f) * 255.0f + 0.5f);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

[numthreads(8, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= count)
        return;

    uint texIdx = TexIndices[i];
    uint bufIdx = BufIndices[i];

    float4 texel = TexTable[NonUniformResourceIndex(texIdx)].SampleLevel(Samp, float2(0.5f, 0.5f), 0.0f);
    uint bufWord = BufTable[NonUniformResourceIndex(bufIdx)][0];

    Out[i * 2u] = packRGBA8(texel);
    Out[i * 2u + 1u] = bufWord;
}
