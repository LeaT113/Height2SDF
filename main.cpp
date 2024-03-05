#include <iostream>
#include <filesystem>

#include "src/ImageIO.hpp"
#include "src/SdfGenerator.hpp"
#include "argh.h"


auto constexpr USAGE =
    R"(Usage: height2sdf <heightmap_image> [-o output_file] [-l layers|10]

Options:
    <heightmap_image>           Input heightmap

    -o, --output <output_file>  Output file for the SDF (.tiff)
                                Default is output_sdf.tiff
    -l, --layers <layers>       Number of layers in the Z dimension
                                Default is 10
)";

int main(int argc, char* argv[])
{
    // Parse arguments
    argh::parser cmdl(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
    if (!cmdl(1) || cmdl[{"-h", "--help"}])
    {
        std::cout << USAGE << std::endl;
        return 1;
    }
    std::filesystem::path heightmapPath (cmdl[1]);
    int layers;
    cmdl({"-l", "--layers"}, 10) >> layers;
    std::filesystem::path outputPath (cmdl({"-o", "--output"}, "output_sdf.tiff").str());

    auto heightmap = ImageIO::LoadHeightmap(absolute(heightmapPath));
    auto sdf = SdfGenerator::GenerateSdfFromHeightmap(heightmap, layers);

    // Convert for export
    Image3D<uint8_t> sdfImg(sdf.Width(), sdf.Height(), sdf.Depth());
    for(int z = 0; z < sdfImg.Depth(); z++)
        for(int y = 0; y < sdfImg.Height(); y++)
            for(int x = 0; x < sdfImg.Width(); x++)
                sdfImg(x, y, z) = static_cast<uint8_t>(sdf(x, y, z) * 255);

    // Export to TIFF
    ImageIO::ExportImage3DToTiff(sdfImg, absolute(outputPath));

    return 0;
}
