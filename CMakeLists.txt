cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES "native")
project(height2sdf LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_ARCHITECTURES 75-real)

add_executable(${PROJECT_NAME} main.cpp
        src/Image/ImageIO.hpp src/Image/ImageIO.cpp
        src/Image/Image2D.hpp src/Image/Image2D.cpp
        src/Image/Image3D.hpp src/Image/Image3D.cpp
        src/SdfGenerator/SdfGenerator.hpp src/SdfGenerator/SdfGenerator.cpp
        src/SdfGenerator/SdfGeneratorCuda.cu
        src/SdfGenerator/SdfGeneratorCuda.cuh
        src/SdfGenerator/ISdfGenerator.cpp
        src/SdfGenerator/ISdfGenerator.h
)

find_package(Stb REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})

find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

find_package(TIFF REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE TIFF::TIFF)
