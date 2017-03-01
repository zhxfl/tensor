#include "tensor.hpp"
using namespace tensor;
int main()
{
    Tensor<float> t(-1, Size(3,2,2));
    for(size_t i = 0; i < t.size().count(); i++){
        t.mutable_cpu_data()[i] = i;
    }

    Tensor<float>t1 = t.slice_from(Offset(0, 1, 1), Size(3, 1, 1));
    for(size_t i = 0; i < t1.size().count(); i++){
        printf("%f ", t1.mutable_cpu_data()[i]);
    }
    printf("\n");

    for(size_t i = 0; i < 2; i++){
        for(size_t j = 0; j < 2; j++){
            for(size_t k = 0; k < 3; k++){
                printf("%f ", t.mutable_cpu_data(Offset(k, j, i))[0]);
            }
        }
    }
    printf("\n");

    for(size_t i = 0; i < 1; i++){
        for(size_t j = 0; j < 1; j++){
            for(size_t k = 0; k < 3; k++){
                printf("%f ", t1.mutable_cpu_data(Offset(k, j, i))[0]);
            }
        }
    }
    
   printf("\n");
    Tensor<float>t2 = t1.slice_from(Offset(1, 0, 0), Size(2, 1, 1));
    for(size_t i = 0; i < 1; i++){
        for(size_t j = 0; j < 1; j++){
            for(size_t k = 0; k < 2; k++){
                printf("%f ", t2.mutable_cpu_data(Offset(k, j, i))[0]);
            }
        }
    }
     
    return 0;
}
