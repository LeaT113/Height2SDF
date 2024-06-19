#include <iostream>
#include <filesystem>
#include <format>
#include <argh.h>

#include "Image/ImageIO.hpp"
#include "SdfGenerator/SdfGenerator.hpp"
#include "SdfGenerator/SdfGeneratorCuda.cuh"


auto constexpr USAGE =
    R"(Usage: height2sdf <heightmap_image> [-o output_file] [-l layers|10]

Options:
    <heightmap_image>           Input heightmap

    -o, --output <output_file>  Output file for the SDF (.tiff)
                                Default is output_sdf.tiff
    -l, --layers <layers>       Number of layers in the Z dimension
                                Default is 10
    -c, --cuda                  Use CUDA for calculation
    -a, --algorithm <index>     0 - JFA (default)
                                1 - 1+JFA
                                2 - 1+JFA+2
                                3 - JFA^2
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
    std::filesystem::path outputPath (cmdl({"-o", "--output"}, "output_sdf.tiff").str());
    int layers;
    cmdl({"-l", "--layers"}, 10) >> layers;
    int algoInt;
    cmdl({"-a", "--algorithm"}, 0) >> algoInt;
    auto algo = static_cast<JfaAlgorithm>(algoInt);
    bool useCuda = cmdl[{"-c", "--cuda"}];

    // Load heightmap
    auto heightmap = ImageIO::LoadHeightmap(absolute(heightmapPath));

    // Run SDF generation
    auto start = std::chrono::high_resolution_clock::now();
    auto sdf = useCuda ?
                SdfGeneratorCuda::GenerateSdfFromHeightmap(heightmap, layers, algo) :
                SdfGenerator::GenerateSdfFromHeightmap(heightmap, layers, algo);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::format("Calculated SDF[{}, {}, {}] in {:.3f}s", sdf.Width(), sdf.Height(), sdf.Depth(),
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0) << std::endl;

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
