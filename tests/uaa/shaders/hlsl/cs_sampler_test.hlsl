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

// Samples a 2D texture via SampleLevel or SampleGrad and stores texels to a buffer.

[[vk::binding(0, 0)]] Texture2D<float4> Tex : register(t0, space0);
[[vk::binding(1, 0)]] SamplerState Samp : register(s1, space0);
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> Out : register(u2, space0);
[[vk::binding(3, 0)]] cbuffer Params : register(b3, space0)
{
    float2 uvOrigin;
    float2 uvStep;
    float lod;
    uint width;
    uint height;
    uint mode;
};

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= width || DTid.y >= height)
        return;

    float2 uv = uvOrigin + float2(DTid.xy) * uvStep;
    float4 sampleVal = (mode == 0u)
        ? Tex.SampleLevel(Samp, uv, lod)
        : Tex.SampleGrad(Samp, uv, float2(uvStep.x, 0.0f), float2(0.0f, uvStep.y));

    Out[DTid.y * width + DTid.x] = sampleVal;
}
