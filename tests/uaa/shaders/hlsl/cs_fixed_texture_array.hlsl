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

[[vk::binding(0, 0)]] Texture2D<float4> Images[4] : register(t0, space0);
[[vk::binding(0, 1)]] RWStructuredBuffer<uint> Out : register(u0, space2);

uint packColour(float4 colour)
{
    uint4 c = uint4(colour * 255.0f + 0.5f);
    return c.x | (c.y << 8) | (c.z << 16) | (c.w << 24);
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    Out[0] = packColour(Images[0].Load(int3(0, 0, 0)));
    Out[1] = packColour(Images[1].Load(int3(0, 0, 0)));
    Out[2] = packColour(Images[2].Load(int3(0, 0, 0)));
    Out[3] = packColour(Images[3].Load(int3(0, 0, 0)));
}
