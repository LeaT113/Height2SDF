#include "SdfGenerator.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

void JfaSeed(const Image2D<float>& heightmap, Image3D<glm::vec3>& out)
{
    for(int z = 0; z < out.Depth(); z++)
        for(int y = 0; y < out.Height(); y++)
            for(int x = 0; x < out.Width(); x++)
            {
                bool isSeed = z >= static_cast<int>((1.0f - heightmap(x, y)) * out.Depth());
                auto uvw = glm::vec3(x, y, static_cast<float>(z) * out.Width() / out.Depth());

                out(x, y, z) = isSeed ? uvw : glm::vec3(0);
            }
}

void JfaStep(const Image3D<glm::vec3>& uvwIn, Image3D<glm::vec3>& uvwOut, int radius)
{
    for(int z = 0; z < uvwIn.Depth(); z++)
        for(int y = 0; y < uvwIn.Height(); y++)
            for(int x = 0; x < uvwIn.Width(); x++)
            {
                auto pixel = glm::vec3(x, y, static_cast<float>(z) * uvwIn.Width() / uvwIn.Depth());
                auto pixelSeed = uvwIn(x, y, z);

                // Search neighbours in cross pattern
                for (int i = -1; i <= 1; i++)
                    for (int j = -1; j <= 1; j++)
                        for (int k = -1; k <= 1; k++)
                        {
                            auto xn = x + i*radius;
                            auto yn = y + j*radius;
                            auto zn = z + k*radius;
                            if (xn < 0 || xn >= uvwIn.Width() ||
                                yn < 0 || yn >= uvwIn.Height() ||
                                zn < 0 || zn >= uvwIn.Depth())
                                continue;

                            const auto& neighbourSeed = uvwIn(xn, yn, zn);

                            bool pixelSeedUndefined = all(equal(pixelSeed, glm::vec3(0)));
                            bool neighbourSeedUndefined = all(equal(neighbourSeed, glm::vec3(0)));

                            if (!neighbourSeedUndefined)
                            {
                                // P takes value of N if empty
                                if (pixelSeedUndefined)
                                    pixelSeed = neighbourSeed;
                                else if (length2(pixel - neighbourSeed) < length2(pixel - pixelSeed))
                                {
                                    pixelSeed = neighbourSeed;
                                }
                            }

                            uvwOut(x, y, z) = pixelSeed;
                        }
            }
}

void JfaDist(const Image3D<glm::vec3>& uvw, Image3D<float>& sdf)
{
    for(int z = 0; z < uvw.Depth(); z++)
        for(int y = 0; y < uvw.Height(); y++)
            for(int x = 0; x < uvw.Width(); x++)
            {
                auto pixel = glm::vec3(x, y, static_cast<float>(z) * uvw.Width() / uvw.Depth());
                const auto& pixelSeed = uvw(x, y, z);

                sdf(x, y, z) = length(pixel - pixelSeed) / uvw.Width();
            }
}

Image3D<float> SdfGenerator::GenerateSdfFromHeightmap(const Image2D<float>& heightmap, int depth, JfaAlgorithm algorithm)
{
    Image3D<glm::vec3> uvw1(heightmap.Width(), heightmap.Height(), depth);
    Image3D<glm::vec3> uvw2(heightmap.Width(), heightmap.Height(), depth);

    // Seed
    JfaSeed(heightmap, uvw1);

    // Step
    if (algorithm == OnePlusJFA || algorithm == OnePlusJFAPlusTwo)
    {
        JfaStep(uvw1, uvw2, 1);
        std::swap(uvw1, uvw2);
    }
    int radius = uvw1.Width();
    while(true)
    {
        radius /= 2;
        JfaStep(uvw1, uvw2, radius);
        std::swap(uvw1, uvw2);

        if(radius <= 1)
            break;
    }
    if (algorithm == OnePlusJFAPlusTwo)
    {
        JfaStep(uvw1, uvw2, 2);
        std::swap(uvw1, uvw2);
        JfaStep(uvw1, uvw2, 1);
        std::swap(uvw1, uvw2);
    }
    else if (algorithm == JFASquared)
    {
        radius = uvw1.Width();
        while(true)
        {
            radius /= 2;
            JfaStep(uvw1, uvw2, radius);
            std::swap(uvw1, uvw2);

            if(radius <= 1)
                break;
        }
    }

    // Distance
    Image3D<float> sdf(uvw1.Width(), uvw1.Height(), uvw1.Depth());
    JfaDist(uvw1, sdf);

    return sdf;
}
