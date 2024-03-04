cmake_minimum_required(VERSION 3.28)
project(Height2SDF)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cpp
        src/ImageIO.hpp src/ImageIO.cpp
        src/Image2D.hpp src/Image2D.cpp
        src/Image3D.hpp src/Image3D.cpp
        src/SdfGenerator.cpp
        src/SdfGenerator.hpp
)

find_package(Stb REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})

find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

find_package(TIFF REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE TIFF::TIFF)
