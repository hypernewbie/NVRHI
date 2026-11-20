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

// Tests comparison sampling against a depth texture using SamplerComparisonState.

[[vk::binding(0, 0)]] Texture2D<float> DepthTex : register(t0, space0);
[[vk::binding(1, 0)]] SamplerComparisonState CmpSamp : register(s1, space0);
[[vk::binding(2, 0)]] RWStructuredBuffer<float> Out : register(u2, space0);
[[vk::binding(3, 0)]] cbuffer Params : register(b3, space0)
{
    float refDepth;
    float uStart;
    float uStep;
    float vCoord;
    uint count;
    uint p0;
    uint p1;
    uint p2;
};

[numthreads(8, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i < count)
    {
        float u = uStart + float(i) * uStep;
        Out[i] = DepthTex.SampleCmpLevelZero(CmpSamp, float2(u, vCoord), refDepth);
    }
}
