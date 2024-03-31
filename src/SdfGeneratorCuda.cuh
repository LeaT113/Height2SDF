#ifndef SDFGENERATORCUDA_CUH
#define SDFGENERATORCUDA_CUH
#include <cstdio>

#include "Image2D.hpp"
#include "Image3D.hpp"


class SdfGeneratorCuda
{
public:
    static Image3D<float> GenerateSdfFromHeightmap(const Image2D<float>& heightmap, int depth);
};



#endif
