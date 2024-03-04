#include <iostream>
#include <filesystem>
#include "src/ImageIO.hpp"
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/io.hpp>

#include "src/SdfGenerator.hpp"

int main(int argc, char* argv[])
{
    if (argc < 2)
        return 1;

    std::filesystem::path heightmapPath (argv[1]);

    auto heightmap = ImageIO::LoadHeightmap(absolute(heightmapPath));

    // Seed Image3D based on heightmap (_ -> glm::vec3 UVW)
    Image3D<glm::vec3> jfa(heightmap.Width(), heightmap.Height(), 10);
    SdfGenerator::JfaSeed(heightmap, jfa);

    // Step Image3D log2 res times (glm::vec2 UV -> glm::vec2 UV)
    int radius = jfa.Width();
    while(true)
    {
        radius /= 2;
        SdfGenerator::JfaStep(jfa, radius);

        if(radius <= 1)
            break;
    }

    // Distance pass Image3D (glm::vec2 UV -> float DIST)
    Image3D<float> sdf(jfa.Width(), jfa.Height(), jfa.Depth());
    SdfGenerator::JfaDist(jfa, sdf);

    // Convert for export (float DIST -> uint8_t DIST)
    Image3D<uint8_t> sdfImg(sdf.Width(), sdf.Height(), sdf.Depth());
    for(int z = 0; z < sdfImg.Depth(); z++)
        for(int y = 0; y < sdfImg.Height(); y++)
            for(int x = 0; x < sdfImg.Width(); x++)
            {
                sdfImg(x, y, z) = static_cast<uint8_t>(sdf(x, y, z) * 255);
            }

    // Export to TIFF
    std::filesystem::path exportPath = "exportTiff.tiff";
    ImageIO::ExportImage3DToTiff(sdfImg, absolute(exportPath));

    return 0;
}
