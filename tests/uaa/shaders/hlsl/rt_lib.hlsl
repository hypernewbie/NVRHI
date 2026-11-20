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

// Ray-tracing shader library exporting RayGen, Miss, ClosestHit, and AnyHit.

[[vk::binding(0, 0)]] RaytracingAccelerationStructure Scene : register(t0, space0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Out : register(u1, space0);

struct RayPayload
{
    uint hit;
    uint instanceID;
    uint primitiveIndex;
    uint rayT;
};

[shader("raygeneration")]
void RayGen()
{
    RayDesc ray;
    if (Out[7] != 0u)
    {
        ray.Origin = float3(asfloat(Out[4]), asfloat(Out[5]), asfloat(Out[6]));
        ray.Direction = float3(asfloat(Out[8]), asfloat(Out[9]), asfloat(Out[10]));
    }
    else
    {
        ray.Origin = float3(0.0f, 0.0f, -2.0f);
        ray.Direction = float3(0.0f, 0.0f, 1.0f);
    }
    ray.TMin = 0.001f;
    ray.TMax = 1000.0f;

    RayPayload payload;
    payload.hit = 0u;
    payload.instanceID = 0u;
    payload.primitiveIndex = 0u;
    payload.rayT = 0u;

    TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

    Out[0] = payload.hit;
    Out[1] = payload.instanceID;
    Out[2] = payload.primitiveIndex;
    Out[3] = payload.rayT;
}

[shader("miss")]
void Miss(inout RayPayload payload)
{
    payload.hit = 0u;
    payload.instanceID = 0u;
    payload.primitiveIndex = 0u;
    payload.rayT = 0u;
}

[shader("closesthit")]
void ClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.hit = 1u;
    payload.instanceID = InstanceID();
    payload.primitiveIndex = PrimitiveIndex();
    payload.rayT = asuint(RayTCurrent());
}

[shader("anyhit")]
void AnyHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    if (InstanceID() == 999u)
    {
        IgnoreHit();
    }
}
