//
// Created by tomasweiss on 3/30/24.
//

#include "SdfGeneratorCuda.cuh"
#include <vector_types.h>

#define xyzToIdx(xyz, dims) (xyz.x + xyz.y * dims.x + xyz.z * dims.x * dims.y)
#define xyzInDims(xyz, dims) (xyz.x < dims.x && xyz.y < dims.y && xyz.z < dims.z)
#define xyzIsZero(xyz) (xyz.x == 0 && xyz.y == 0 && xyz.z == 0)
#define xyzDist2(xyz1, xyz2) ((xyz1.x - xyz2.x)*(xyz1.x - xyz2.x) + (xyz1.y - xyz2.y)*(xyz1.y - xyz2.y) + (xyz1.z - xyz2.z)*(xyz1.z - xyz2.z))

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void hello(float* fl) {
    printf("Hello world! %f\n", *fl);
}

__global__ void JfaSeed(const float* heightmap, uint3* uvw, uint3 dims)
{
    uint3 pos = make_uint3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z);

    uint heightmapIdx = xyzToIdx(make_uint3(pos.x, pos.y, 0), dims);
    uint uvwIdx = xyzToIdx(pos, dims);

    //if(pos.z >= (uint)((1.0f - heightmap[heightmapIdx]) * (float)(dims.z-1)))
    if((float)pos.z / (float)dims.z > 1.0 - heightmap[heightmapIdx])
    {
        uvw[uvwIdx] = pos;
    }
    else
    {
        uvw[uvwIdx] = make_uint3(0, 0, 0);
    }
}

__global__ void JfaStep(uint3* uvwIn, uint3* uvwOut, uint3 dims, uint stepSize)
{
    uint3 pos = make_uint3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z);

    uint3 pixel = pos;
    uint3 pixelSeed = uvwIn[xyzToIdx(pos, dims)];

    // TODO Unroll?
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            for (int k = -1; k <= 1; k++)
            {
                uint3 neighbourPos = make_uint3(pos.x + i * stepSize, pos.y + j * stepSize, pos.z + k * stepSize);
                if (!xyzInDims(neighbourPos, dims))
                    continue;

                uint3 neighbourSeed = uvwIn[xyzToIdx(neighbourPos, dims)];
                int3 neighbourSeedI = make_int3(neighbourSeed.x, neighbourSeed.y, neighbourSeed.z);

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

__global__ void JfaDist(const uint3* uvw, float* sdf, uint3 dims)
{
    uint3 pos = make_uint3(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z);

    uint idx = xyzToIdx(pos, dims);

    float3 pixel = make_float3(pos.x, pos.y, pos.z);
    uint3 pixelSeed = uvw[idx];
    float3 pixelSeedF = make_float3(pixelSeed.x, pixelSeed.y, pixelSeed.z);

    sdf[idx] = norm3df(pixel.x - pixelSeedF.x, pixel.y - pixelSeedF.y, pixel.z - pixelSeedF.z) / 16;
}

Image3D<float> SdfGeneratorCuda::GenerateSdfFromHeightmap(const Image2D<float> &heightmap, int depth)
{
    // GPU memory
    float* heightmapDevice;
    cudaMalloc(&heightmapDevice, heightmap.DataSize());
    cudaMemcpy(heightmapDevice, heightmap.DataPtr(), heightmap.DataSize(), cudaMemcpyHostToDevice);

    uint3* uvwDevice1;
    cudaMalloc(&uvwDevice1, heightmap.Width() * heightmap.Height() * depth * sizeof(uint3));
    uint3* uvwDevice2;
    cudaMalloc(&uvwDevice2, heightmap.Width() * heightmap.Height() * depth * sizeof(uint3));

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
    cudaDeviceSynchronize();

    // Step
    uint stepSize = dims.x;
    while(true)
    {
        stepSize /= 2;
        JfaStep<<<blocks, blockSize>>>(uvwDevice1, uvwDevice2, dims, stepSize);
        cudaDeviceSynchronize();
        std::swap(uvwDevice1, uvwDevice2);

        if(stepSize <= 1)
            break;
    }

    // Distance
    JfaDist<<<blocks, blockSize>>>(uvwDevice1, sdfDevice, dims);
    cudaDeviceSynchronize();

    // Retrieve
    auto sdf = Image3D<float>(128, 128, 16);
    cudaMemcpy(sdf.DataPtr(), sdfDevice, heightmap.Width() * heightmap.Height() * depth * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(heightmapDevice);
    cudaFree(uvwDevice1);
    cudaFree(uvwDevice2);
    cudaFree(sdfDevice);

    return sdf;
}
