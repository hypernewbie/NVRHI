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

// The vertex stage reads element 0 of a texture array of the bindless set at
// binding 0 and passes its colour through. Paired with ps_bindless_stage.hlsl,
// whose fragment array sits at binding 1, so each stage has its own array.

[[vk::binding(0, 0)]] Texture2D<float4> VertTable[] : register(t0, space0);
[[vk::binding(1, 1)]] SamplerState Samp : register(s1, space2);

struct VSOut
{
    float4 pos : SV_Position;
    float4 col : COLOR0;
};

VSOut main(uint id : SV_VertexID)
{
    static const float2 kPositions[3] =
    {
        float2(-1.0f, -1.0f),
        float2(3.0f, -1.0f),
        float2(-1.0f, 3.0f)
    };

    VSOut o;
    o.pos = float4(kPositions[id], 0.0f, 1.0f);
    o.col = VertTable[0].SampleLevel(Samp, float2(0.25f, 0.25f), 0);
    return o;
}
