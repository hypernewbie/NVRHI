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

// Exercises raw (ByteAddressBuffer) and typed (Buffer<uint>) SRV/UAV combinations.

[[vk::binding(0, 0)]] ByteAddressBuffer RawIn : register(t0, space0);
[[vk::binding(1, 0)]] RWByteAddressBuffer RawOut : register(u1, space0);
[[vk::binding(2, 0)]] Buffer<uint> TypedIn : register(t2, space0);
[[vk::binding(3, 0)]] RWBuffer<uint> TypedOut : register(u3, space0);
[[vk::binding(4, 0)]] cbuffer Params : register(b4, space0)
{
    uint count;
    uint mode;
    uint addVal;
    uint pad;
};

[numthreads(64, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= count)
        return;

    if (mode == 0u)
    {
        RawOut.Store(i * 4u, RawIn.Load(i * 4u) + addVal);
    }
    else if (mode == 1u)
    {
        TypedOut[i] = RawIn.Load(i * 4u) + addVal;
    }
    else if (mode == 2u)
    {
        RawOut.Store(i * 4u, TypedIn[i] + addVal);
    }
    else
    {
        TypedOut[i] = TypedIn[i] + addVal;
    }
}
