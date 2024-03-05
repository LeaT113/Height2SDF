#include <iostream>
#include <filesystem>

#include "src/ImageIO.hpp"
#include "src/SdfGenerator.hpp"

int main(int argc, char* argv[])
{
    if (argc < 2)
        return 1;

    std::filesystem::path heightmapPath (argv[1]);

    auto heightmap = ImageIO::LoadHeightmap(absolute(heightmapPath));

    auto sdf = SdfGenerator::GenerateSdfFromHeightmap(heightmap, 10);

    // Convert for export
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
