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

// Mesh shader emitting a single screen-covering triangle tinted from an amplification payload.

struct Payload
{
    float4 color;
};

struct MeshVertex
{
    float4 pos : SV_Position;
    float4 col : COLOR0;
};

[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void main(in payload Payload p, out vertices MeshVertex verts[3], out indices uint3 tris[1])
{
    SetMeshOutputCounts(3, 1);

    verts[0].pos = float4(-1.0f, -1.0f, 0.5f, 1.0f);
    verts[0].col = p.color;

    verts[1].pos = float4(3.0f, -1.0f, 0.5f, 1.0f);
    verts[1].col = p.color;

    verts[2].pos = float4(-1.0f, 3.0f, 0.5f, 1.0f);
    verts[2].col = p.color;

    tris[0] = uint3(0u, 1u, 2u);
}
