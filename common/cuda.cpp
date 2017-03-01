//by zhxfl 2016.03.01
//copy from purine2
#include "cuda.hpp"

namespace tensor{

CUDA& cuda() {
    //static thread_local CUDA cuda;
    static CUDA cuda;
    return cuda;
}

cudaStream_t stream() {
    return cuda().stream();
}

}
