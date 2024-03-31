#ifndef IMAGE3D_HPP
#define IMAGE3D_HPP
#include <vector>


template <typename T>
class Image3D
{
public:
    Image3D(size_t width, size_t height, size_t depth);

    const T& operator ()(int x, int y, int z) const;
    T& operator ()(int x, int y, int z);

    const T* GetPixelsAt(size_t x, size_t y, size_t z) const;

    T* DataPtr();
    size_t DataSize() const;

    size_t Width() const;
    size_t Height() const;
    size_t Depth() const;

private:
    size_t _width, _height, _depth;
    std::vector<T> _data;
};


template <typename T>
Image3D<T>::Image3D(size_t width, size_t height, size_t depth)
    : _width(width), _height(height), _depth(depth), _data(width * height * depth)
{}

template <typename T>
const T& Image3D<T>::operator()(int x, int y, int z) const
{
    return _data[x + y * _width + z * _width * _height];
}

template <typename T>
T& Image3D<T>::operator()(int x, int y, int z)
{
    return _data[x + y * _width + z * _width * _height];
}

template <typename T>
const T* Image3D<T>::GetPixelsAt(size_t x, size_t y, size_t z) const
{
    return _data.data() + x + y * _width + z * _width * _height;
}

template<typename T>
T * Image3D<T>::DataPtr()
{
    return _data.data();
}

template<typename T>
size_t Image3D<T>::DataSize() const
{
    return _data.size() * sizeof(T);
}

template <typename T>
size_t Image3D<T>::Width() const
{
    return _width;
}

template <typename T>
size_t Image3D<T>::Height() const
{
    return _height;
}

template <typename T>
size_t Image3D<T>::Depth() const
{
    return _depth;
}



#endif
