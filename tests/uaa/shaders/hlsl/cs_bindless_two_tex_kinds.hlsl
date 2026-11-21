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

// Two read-only arrays of the bindless set that both hold textures on Metal: a
// sampled texture array and a typed buffer array. The outputs combine one
// element of each, so a backend that confuses the two arrays fails.

[[vk::binding(0, 0)]] Texture2D<float4> TexTable[] : register(t0, space0);
[[vk::binding(1, 0)]] Buffer<uint4> TypedTable[] : register(t0, space1);

[[vk::binding(0, 1)]] StructuredBuffer<uint> TexIndices : register(t0, space2);
[[vk::binding(1, 1)]] StructuredBuffer<uint> TypedIndices : register(t1, space2);
[[vk::binding(2, 1)]] RWStructuredBuffer<uint4> Out : register(u2, space2);
[[vk::binding(3, 1)]] SamplerState Samp : register(s3, space2);
[[vk::binding(4, 1)]] cbuffer Params : register(b4, space2)
{
    uint count;
};

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= count)
        return;

    const float4 texel = TexTable[TexIndices[id.x]].SampleLevel(Samp, float2(0.25f, 0.25f), 0);
    const uint4 typed = TypedTable[TypedIndices[id.x]].Load(0);

    Out[id.x] = uint4(texel * 255.0f + 0.5f) + typed;
}
