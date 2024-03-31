#ifndef IMAGEIO_HPP
#define IMAGEIO_HPP
#include <filesystem>

#include "Image2D.hpp"
#include "Image3D.hpp"

class ImageIO
{
public:
    static Image2D<float> LoadHeightmap(const std::filesystem::path& path);

    static void ExportImage3DToTiff(const Image3D<uint8_t>& image, const std::filesystem::path& path);
};

#endif
