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

// Emits a single fullscreen or scaled triangle from SV_VertexID using push constants.

struct PC
{
    float4 color;
    float depth;
    float scale;
    float2 offset;
};

[[vk::push_constant]] ConstantBuffer<PC> g_PC : register(b0, space0);

float4 main(uint vid : SV_VertexID) : SV_Position
{
    float2 base = float2((vid == 1u) ? 3.0f : -1.0f, (vid == 2u) ? 3.0f : -1.0f);
    float s = (g_PC.scale == 0.0f) ? 1.0f : g_PC.scale;
    return float4(base * s + g_PC.offset, g_PC.depth, 1.0f);
}
