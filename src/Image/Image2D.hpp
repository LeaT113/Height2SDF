#ifndef IMAGE_HPP
#define IMAGE_HPP
#include <vector>


template <typename T>
class Image2D
{
public:
    Image2D(size_t width, size_t height);

    const T& operator ()(int x, int y) const;
    T& operator ()(int x, int y);

    const T* GetPixelsAt(size_t x, size_t y) const;

    const T* DataPtr() const;
    T* DataPtr();
    size_t DataSize() const;

    size_t Width() const;
    size_t Height() const;

private:
    size_t _width, _height;
    std::vector<T> _data;
};


template <typename T>
Image2D<T>::Image2D(size_t width, size_t height)
    : _width(width), _height(height), _data(width * height)
{}

template <typename T>
const T& Image2D<T>::operator()(int x, int y) const
{
    return _data[x + y * _width];
}

template <typename T>
T& Image2D<T>::operator()(int x, int y)
{
    return _data[x + y * _width];
}

template <typename T>
const T* Image2D<T>::GetPixelsAt(size_t x, size_t y) const
{
    return _data.data() + x + y * _width;
}

template<typename T>
const T * Image2D<T>::DataPtr() const
{
    return _data.data();
}

template <typename T>
T* Image2D<T>::DataPtr()
{
    return _data.data();
}

template<typename T>
size_t Image2D<T>::DataSize() const
{
    return _width * _height * sizeof(T);
}

template <typename T>
size_t Image2D<T>::Width() const
{
    return _width;
}

template <typename T>
size_t Image2D<T>::Height() const
{
    return _height;
}

#endif
