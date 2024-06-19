//
// Created by tomasweiss on 3/30/24.
//

#include "SdfGeneratorCuda.cuh"
#include <vector_types.h>
#include "ISdfGenerator.h"

#define xyzToIdx(xyz, dims) (xyz.x + xyz.y * dims.x + xyz.z * dims.x * dims.y)
#define xyzInDims(xyz, dims) (xyz.x < dims.x && xyz.y < dims.y && xyz.z < dims.z)
#define xyzIsZero(xyz) (xyz.x == 0 && xyz.y == 0 && xyz.z == 0)
#define xyzDist2(xyz1, xyz2) ((xyz1.x - xyz2.x)*(xyz1.x - xyz2.x) + (xyz1.y - xyz2.y)*(xyz1.y - xyz2.y) + (xyz1.z - xyz2.z)*(xyz1.z - xyz2.z))
#define posToUvw(pos, dims) (make_float3(pos.x, pos.y, (float)pos.z * dims.x / dims.z))


__global__ void JfaSeed(const float* heightmap, float3* uvw, uint3 dims)
{
    uint3 pos = make_uint3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z);

    uint heightmapIdx = xyzToIdx(make_uint3(pos.x, pos.y, 0), dims);
    uint uvwIdx = xyzToIdx(pos, dims);

    if(pos.z >= (uint)((1.0f - heightmap[heightmapIdx]) * dims.z))
    {
        uvw[uvwIdx] = posToUvw(pos, dims);
    }
    else
    {
        uvw[uvwIdx] = make_float3(0, 0, 0);
    }
}

__global__ void JfaStep(float3* uvwIn, float3* uvwOut, uint3 dims, uint stepSize)
{
    uint3 pos = make_uint3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z);

    float3 pixel = posToUvw(pos, dims);
    float3 pixelSeed = uvwIn[xyzToIdx(pos, dims)];

    // TODO Unroll?
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
            {
                uint3 neighbourPos = make_uint3(pos.x + i * stepSize, pos.y + j * stepSize, pos.z + k * stepSize);
                if (!xyzInDims(neighbourPos, dims))
                    continue;

                float3 neighbourSeed = uvwIn[xyzToIdx(neighbourPos, dims)];

                bool pixelUndefined = xyzIsZero(pixelSeed);
                bool neighbourUndefined = xyzIsZero(neighbourSeed);

                if(!neighbourUndefined)
                {
                    // P takes value of N if empty
                    if (pixelUndefined)
                        pixelSeed = neighbourSeed;
                    // P takes value of N if it's seed is closer
                    else if (xyzDist2(pixel, neighbourSeed) < xyzDist2(pixel, pixelSeed))
                    {
                        pixelSeed = neighbourSeed;
                    }
                }
            }

    uvwOut[xyzToIdx(pos, dims)] = pixelSeed;
}

__global__ void JfaDist(const float3* uvw, float* sdf, uint3 dims)
{
    uint3 pos = make_uint3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z);

    uint idx = xyzToIdx(pos, dims);

    float3 pixel = posToUvw(pos, dims);
    float3 pixelSeed = uvw[idx];

    sdf[idx] = norm3df(pixel.x - pixelSeed.x, pixel.y - pixelSeed.y, pixel.z - pixelSeed.z) / dims.x;
}

Image3D<float> SdfGeneratorCuda::GenerateSdfFromHeightmap(const Image2D<float> &heightmap, int depth, JfaAlgorithm algorithm)
{
    // GPU memory
    float* heightmapDevice;
    cudaMalloc(&heightmapDevice, heightmap.DataSize());
    cudaMemcpy(heightmapDevice, heightmap.DataPtr(), heightmap.DataSize(), cudaMemcpyHostToDevice);

    float3* uvwDevice1;
    cudaMalloc(&uvwDevice1, heightmap.Width() * heightmap.Height() * depth * sizeof(float3));
    float3* uvwDevice2;
    cudaMalloc(&uvwDevice2, heightmap.Width() * heightmap.Height() * depth * sizeof(float3));

    float* sdfDevice;
    cudaMalloc(&sdfDevice, heightmap.Width() * heightmap.Height() * depth * sizeof(float));

    // Dimensions
    uint3 dims = make_uint3(heightmap.Width(), heightmap.Height(), depth);
    dim3 blockSize(16, 16, 4);
    dim3 blocks(
        ceil(dims.x / (float)blockSize.x),
        ceil(dims.y / (float)blockSize.y),
        ceil(dims.z / (float)blockSize.z)
        );

    // Seed
    JfaSeed<<<blocks, blockSize>>>(heightmapDevice, uvwDevice1, dims);

    // Step
    if (algorithm == OnePlusJFA || algorithm == OnePlusJFAPlusTwo)
    {
        JfaStep<<<blocks, blockSize>>>(uvwDevice1, uvwDevice2, dims, 1);
        std::swap(uvwDevice1, uvwDevice2);
    }
    uint stepSize = dims.x;
    while(true)
    {
        stepSize /= 2;
        JfaStep<<<blocks, blockSize>>>(uvwDevice1, uvwDevice2, dims, stepSize);
        std::swap(uvwDevice1, uvwDevice2);

        if(stepSize <= 1)
            break;
    }
    if(algorithm == OnePlusJFAPlusTwo)
    {
        JfaStep<<<blocks, blockSize>>>(uvwDevice1, uvwDevice2, dims, 2);
        std::swap(uvwDevice1, uvwDevice2);
        JfaStep<<<blocks, blockSize>>>(uvwDevice1, uvwDevice2, dims, 1);
        std::swap(uvwDevice1, uvwDevice2);
    }
    else if (algorithm == JFASquared)
    {
        stepSize = dims.x;
        while(true)
        {
            stepSize /= 2;
            JfaStep<<<blocks, blockSize>>>(uvwDevice1, uvwDevice2, dims, stepSize);
            std::swap(uvwDevice1, uvwDevice2);

            if(stepSize <= 1)
                break;
        }
    }

    // Distance
    JfaDist<<<blocks, blockSize>>>(uvwDevice1, sdfDevice, dims);

    // Retrieve
    auto sdf = Image3D<float>(dims.x, dims.y, dims.z);
    cudaMemcpy(sdf.DataPtr(), sdfDevice, heightmap.Width() * heightmap.Height() * depth * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(heightmapDevice);
    cudaFree(uvwDevice1);
    cudaFree(uvwDevice2);
    cudaFree(sdfDevice);

    return sdf;
}
