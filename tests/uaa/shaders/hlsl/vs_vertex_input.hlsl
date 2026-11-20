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

// Vertex shader testing per-vertex and per-instance input attributes plus set 0 CB.

[[vk::binding(0, 0)]] cbuffer VsCB : register(b0, space0)
{
    float4 posScale;
};

struct VSInput
{
    float3 pos : POSITION;
    float4 col0 : COLOR0;
    float2 instOffset : TEXCOORD0;
    float4 instTint : COLOR1;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float4 col : COLOR0;
};

VSOutput main(VSInput input)
{
    VSOutput o;
    o.pos = float4(input.pos.xy * posScale.xy + input.instOffset, input.pos.z * posScale.z, 1.0f);
    o.col = input.col0 * input.instTint;
    return o;
}
