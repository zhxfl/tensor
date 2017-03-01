//by zhxfl 2017.02.28
#include "tensor.hpp"
#include "size.hpp"

namespace tensor{


template<typename T>
Tensor<T>::Tensor(int device, Size& size, Offset& offset,
        Stride& stride) : _size(size), _offset(offset), _stride(stride), _device(device) {
    }

template<typename T>
Tensor<T>::Tensor(int device, Size size): _size(size), _device(device), _offset(std::vector<int>(size.size(), 0)), _stride(size) {
    }

template<typename T>
Tensor<T>::~Tensor() {
    _data.reset();
}

template<typename T>
int Tensor<T>::offset(Offset off, Stride& stride) {
    int ret = 0;
    for(size_t i = 0; i < off.size(); i++){
        ret += off[i] * stride[i];
    }
    return ret;
}

template<typename T>
void Tensor<T>::alloc_mem(T** data,Size& size,int device) {
    CHECK_GT(size.count(), 0);
    if (device < 0) {
        //cudaHostAlloc(data, sizeof(T) * size.count(), cudaHostAllocPortable);
        *data = (T*) malloc(sizeof(T) * size.count());
    } else {
        //SWITCH_DEVICE(device);
        // #ifndef NDEBUG
        //     CUDA_CHECK(cudaMalloc(data, sizeof(T) * (1 + size.count())));
        // #else
        //CUDA_CHECK(cudaMalloc(data, sizeof(T) * size.count()));
        // #endif
        //SWITCH_BACK(device);
    }
}

template<typename T>
void Tensor<T>::free_mem(T* data, int device) {
    if (data == NULL) {
        return;
    }
    if (device < 0) {
        free(data);
        // cudaFreeHost(data);
    } else {
        //SWITCH_DEVICE(device);
        //CUDA_CHECK(cudaFree(data));
        //SWITCH_BACK(device);
    }
}

template<typename T>
Tensor<T> Tensor<T>::slice_from(Offset off, Size size) {
    Offset tmp = _offset + off;
    Tensor other(_device, size, tmp, _stride);
    other._data = _data;
    return other;
}

template<typename T>
void Tensor<T>::delete_data() {
    _data.reset();
}

template<typename T>
 T* Tensor<T>::data()  {
    CHECK(_data);
    return _data.get() + Tensor<T>::offset(_offset, _stride);
}

template<typename T>
T* Tensor<T>::mutable_data() {
    if (!_data) {
        CHECK(is_contiguous());
        T* ptr;
        Tensor<T>::alloc_mem(&ptr, _size, _device);
        _data.reset(ptr, bind(Tensor<T>::free_mem, std::placeholders::_1,
                    _device));
    }
    return _data.get() + Tensor<T>::offset(_offset, _stride);
}

template<typename T>
T* Tensor<T>::mutable_data(Offset off){
    if (!_data) {
        CHECK(is_contiguous());
        T* ptr;
        Tensor<T>::alloc_mem(&ptr, _size, _device);
        _data.reset(ptr, bind(Tensor<T>::free_mem, std::placeholders::_1,
                    _device));
    }
    return _data.get() + Tensor<T>::offset(_offset + off, _stride);
}

template<typename T>
bool Tensor<T>::is_contiguous(){
    return Stride(_size) == _stride;
}

template class Tensor<float>;
template class Tensor<int>;
}
