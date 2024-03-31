#ifndef SDFGENERATORCUDA_CUH
#define SDFGENERATORCUDA_CUH

#include "ISdfGenerator.h"
#include "../Image/Image2D.hpp"
#include "../Image/Image3D.hpp"

class SdfGeneratorCuda
{
public:
    static Image3D<float> GenerateSdfFromHeightmap(const Image2D<float>& heightmap, int depth, JfaAlgorithm algorithm);
};



#endif
