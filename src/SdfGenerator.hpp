#ifndef SDFGENERATOR_HPP
#define SDFGENERATOR_HPP
#include <glm/glm.hpp>

#include "Image2D.hpp"
#include "Image3D.hpp"
#include "SdfGenerator.hpp"


class SdfGenerator
{
public:
    static Image3D<float> GenerateSdfFromHeightmap(const Image2D<float>& heightmap, int depth);

private:
    static void JfaSeed(const Image2D<float>& heightmap, Image3D<glm::vec3>& out);
    static void JfaStep(Image3D<glm::vec3>& img, int radius);
    static void JfaDist(const Image3D<glm::vec3>& in, Image3D<float>& out);
};



#endif
