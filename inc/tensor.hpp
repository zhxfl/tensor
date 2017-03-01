//by zhxfl 2017.02.28
#pragma once
#include <memory>
#include "size.hpp"

using std::shared_ptr;

namespace tensor{
template<typename T>
class Tensor {
public:
    explicit Tensor(int device, Size& size,
            Offset& offset, Stride& stride);
    explicit Tensor(int device, Size size);
    virtual ~Tensor();

    inline  Size& size()  { return _size; }
    inline  Stride& stride()  { return _stride; }
    inline  Offset& offset()  { return _offset; }
    inline int device()  { return _device; }

    Tensor<T> slice_from(Offset off, Size size);
    void delete_data();

    inline T* mutable_gpu_data() {
        CHECK(_device >= 0);
        return mutable_data();
    }

    inline T* mutable_cpu_data(Offset off){
        CHECK(_device < 0);
        return mutable_data(off);
    }

    inline  T* gpu_data() {
        CHECK(_device >= 0);
        return data();
    }
    inline T* mutable_cpu_data() {
        CHECK(_device < 0);
        return mutable_data();
    }
    inline  T* cpu_data() {
        CHECK(_device < 0);
        return data();
    }
    T* mutable_data();
    T* mutable_data( Offset off);
    T* data() ;
    bool is_contiguous();

protected:
    Size _size;
    Offset _offset;
    Stride _stride;
    shared_ptr<T> _data;
    int _device;
    // static
    static int offset(Offset off, Stride& stride);
    static void alloc_mem(T** data, Size& size, int device);
    static void free_mem(T* data, int device);
};

}
