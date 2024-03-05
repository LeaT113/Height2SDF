#include "SdfGenerator.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

Image3D<float> SdfGenerator::GenerateSdfFromHeightmap(const Image2D<float>& heightmap, int depth)
{
    Image3D<glm::vec3> jfa(heightmap.Width(), heightmap.Height(), depth);

    // Seed Image3D based on heightmap (_ -> glm::vec3 UVW)
    JfaSeed(heightmap, jfa);

    // Step Image3D log2 res times (glm::vec3 UV -> glm::vec3 UV)
    int radius = jfa.Width();
    while(true)
    {
        radius /= 2;
        JfaStep(jfa, radius);

        if(radius <= 1)
            break;
    }

    // Distance pass Image3D (glm::vec2 UV -> float DIST)
    Image3D<float> sdf(jfa.Width(), jfa.Height(), jfa.Depth());
    JfaDist(jfa, sdf);

    return sdf;
}

void SdfGenerator::JfaSeed(const Image2D<float>& heightmap, Image3D<glm::vec3>& out)
{
    for(int z = 0; z < out.Depth(); z++)
        for(int y = 0; y < out.Height(); y++)
            for(int x = 0; x < out.Width(); x++)
            {
                float pixelDepth = static_cast<float>(z) / out.Depth();
                float heightmapDepth = 1 - heightmap(x, y);
                glm::vec3 uvw (static_cast<float>(x) / out.Width(), static_cast<float>(y) / out.Height(), static_cast<float>(z) / out.Depth());

                out(x, y, z) = pixelDepth > heightmapDepth ? uvw : glm::vec3(0);
            }
}

void SdfGenerator::JfaStep(Image3D<glm::vec3>& img, int radius)
{
    auto imageDimensions = glm::vec3(img.Width(), img.Height(), img.Depth());

    for(int z = 0; z < img.Depth(); z++)
        for(int y = 0; y < img.Height(); y++)
            for(int x = 0; x < img.Width(); x++)
            {
                auto pixel = glm::vec3(x, y, z) / imageDimensions;
                auto& pixelSeed = img(x, y, z);

                // Search neighbours in cross pattern
                for (int i = -1; i <= 1; i++)
                    for (int j = -1; j <= 1; j++)
                        for (int k = -1; k <= 1; k++)
                        {
                            auto xn = x + i*radius;
                            auto yn = y + j*radius;
                            auto zn = z + k*radius;
                            if (xn < 0 || xn >= img.Width() ||
                                yn < 0 || yn >= img.Height() ||
                                zn < 0 || zn >= img.Depth())
                                continue;

                            const auto& neighbourSeed = img(xn, yn, zn);

                            bool pixelSeedUndefined = all(equal(pixelSeed, glm::vec3(0)));
                            bool neighbourSeedUndefined = all(equal(neighbourSeed, glm::vec3(0)));

                            // P takes value of N if empty
                            if (pixelSeedUndefined && !neighbourSeedUndefined)
                                pixelSeed = neighbourSeed;

                            // P takes value of N if it's seed is closer
                            else if (!pixelSeedUndefined && !neighbourSeedUndefined)
                            {
                                auto pixelDist = length2(pixel - pixelSeed);
                                auto neighboursDist = length2(pixel - neighbourSeed);

                                if (neighboursDist < pixelDist)
                                    pixelSeed = neighbourSeed;
                            }
                        }
            }
}

void SdfGenerator::JfaDist(const Image3D<glm::vec3>& in, Image3D<float>& out)
{
    auto imageDimensions = glm::vec3(in.Width(), in.Height(), in.Depth());

    for(int z = 0; z < in.Depth(); z++)
        for(int y = 0; y < in.Height(); y++)
            for(int x = 0; x < in.Width(); x++)
            {
                auto pixel = glm::vec3(x, y, z) / imageDimensions;
                const auto& pixelSeed = in(x, y, z);

                out(x, y, z) = length(pixel - pixelSeed);
            }
}
