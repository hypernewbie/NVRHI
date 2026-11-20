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

// Populates indirect argument buffers on the GPU for dispatchIndirect/drawIndirect tests.

[[vk::binding(0, 0)]] cbuffer Params : register(b0, space0)
{
    uint word0;
    uint word1;
    uint word2;
    uint word3;
    uint word4;
    uint offsetWords;
    uint count;
    uint pad;
};

[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Out : register(u1, space0);

[numthreads(1, 1, 1)]
void main()
{
    uint words[5] = { word0, word1, word2, word3, word4 };
    uint n = min(count, 5u);
    for (uint k = 0u; k < n; ++k)
    {
        Out[offsetWords + k] = words[k];
    }
}
