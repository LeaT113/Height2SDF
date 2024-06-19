#include "ImageIO.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <iostream>
#include <stb_image.h>
#include <tiffio.h>
#include "Image3D.hpp"


Image2D<float> ImageIO::LoadHeightmap(const std::filesystem::path& path)
{
    stbi_set_flip_vertically_on_load(1);

    int width = 0, height = 0, channels = 0;
    uint8_t *img = stbi_load(path.c_str(), &width, &height, &channels, 1);
    if (img == nullptr)
        throw std::runtime_error("Failed to load heightmap");

    auto image2D = Image2D<float>(width, height);
    for(size_t i = 0; i < width * height; i++)
        image2D.DataPtr()[i] = static_cast<float>(img[i]) / 255.0f;

    stbi_image_free(img);

    return image2D;
}

void ImageIO::ExportImage3DToTiff(const Image3D<uint8_t>& image, const std::filesystem::path& path)
{
    TIFF* tif = TIFFOpen(path.c_str(), "w");

    for (size_t layer = 0; layer < image.Depth(); layer++)
    {
        TIFFSetField (tif, TIFFTAG_IMAGEWIDTH, image.Width());
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, image.Height());
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

        for (size_t row = 0; row < image.Height(); row++)
            TIFFWriteScanline(tif, (void*) image.GetPixelsAt(0, row, layer), row, 0);

        if (layer < image.Depth() - 1)
            TIFFWriteDirectory(tif);
    }
}