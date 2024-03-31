#ifndef SDFGENERATOR_HPP
#define SDFGENERATOR_HPP
#include <glm/glm.hpp>

#include "ISdfGenerator.h"
#include "../Image/Image2D.hpp"
#include "../Image/Image3D.hpp"

class SdfGenerator
{
public:
    static Image3D<float> GenerateSdfFromHeightmap(const Image2D<float>& heightmap, int depth, JfaAlgorithm algorithm);
};



#endif
