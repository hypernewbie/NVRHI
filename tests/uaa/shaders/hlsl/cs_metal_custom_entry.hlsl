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

// Compute shader with a custom entry point used as a frozen Metal SPIR-V fixture.

struct PC
{
    uint addVal;
    uint p1;
    uint p2;
    uint p3;
};

[[vk::push_constant]] ConstantBuffer<PC> g_PC : register(b0, space0);
[[vk::binding(0, 0)]] Buffer<uint> TypedIn : register(t0, space0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Out : register(u1, space0);
[[vk::binding(2, 0)]] Texture2D<float4> Tex : register(t2, space0);
[[vk::binding(3, 0)]] SamplerState Samp : register(s3, space0);

uint packRGBA8(float4 c)
{
    uint r = uint(clamp(c.r, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint g = uint(clamp(c.g, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint b = uint(clamp(c.b, 0.0f, 1.0f) * 255.0f + 0.5f);
    uint a = uint(clamp(c.a, 0.0f, 1.0f) * 255.0f + 0.5f);
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

[numthreads(1, 1, 1)]
void uaa_custom_entry()
{
    Out[0] = TypedIn[0] + g_PC.addVal;
    Out[1] = packRGBA8(Tex.SampleLevel(Samp, float2(0.5f, 0.5f), 0.0f));
}
