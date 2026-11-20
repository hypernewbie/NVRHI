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

// Inline ray-query compute shader writing hit/miss, instance ID, primitive index, and hit distance.

[[vk::binding(0, 0)]] RaytracingAccelerationStructure Scene : register(t0, space0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Out : register(u1, space0);
[[vk::binding(2, 0)]] cbuffer Params : register(b2, space0)
{
    float4 origins[8];
    float4 dirs[8];
    uint rayMask;
    uint rayCount;
    uint p0;
    uint p1;
};

[numthreads(8, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    uint i = DTid.x;
    if (i >= rayCount || i >= 8u)
        return;

    RayDesc ray;
    ray.Origin = origins[i].xyz;
    ray.Direction = dirs[i].xyz;
    ray.TMin = 0.001f;
    ray.TMax = 1000.0f;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(Scene, RAY_FLAG_NONE, rayMask, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            q.CommitNonOpaqueTriangleHit();
        }
    }

    uint base = i * 4u;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        Out[base + 0u] = 1u;
        Out[base + 1u] = q.CommittedInstanceID();
        Out[base + 2u] = q.CommittedPrimitiveIndex();
        Out[base + 3u] = asuint(q.CommittedRayT());
    }
    else
    {
        Out[base + 0u] = 0u;
        Out[base + 1u] = 0u;
        Out[base + 2u] = 0u;
        Out[base + 3u] = 0u;
    }
}
