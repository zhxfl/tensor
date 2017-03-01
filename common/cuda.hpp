//by zhxfl 2017.03.01
//copy from purine2
#pragma once

#include <glog/logging.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <driver_types.h>  // cuda driver types

namespace tensor{

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << curandGetErrorString(status); \
  } while (0)

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " " \
      << cudnnGetErrorString(status); \
  } while (0)

/**
 * @def THREAD_SET_CUDA_DEVICE(device_id)
 * @brief set the cuda device of the thread.
 *
 * @param device_id is the device ordinal.
 */
#define THREAD_SET_CUDA_DEVICE(device_id) \
  int device_count = 0; \
  CUDA_CHECK(cudaGetDeviceCount(&device_count)); \
  CHECK_LT(device_id, device_count); \
  CUDA_CHECK(cudaSetDevice(device_id))

/**
 * set cuda device to device_id and keep the original device_id
 * need to call switch back as a pair
 */
#define SWITCH_DEVICE(device_id) \
  int current_device_id_; \
  CUDA_CHECK(cudaGetDevice(&current_device_id_)); \
  if (device_id != current_device_id_) { \
    CUDA_CHECK(cudaSetDevice(device_id)); \
  }
#define SWITCH_BACK(device_id) \
  if (current_device_id_ != device_id) { \
    CUDA_CHECK(cudaSetDevice(current_device_id_)); \
  }

class CUDA {
 private:
  // disable copy and assignment
  CUDA(const CUDA&);
  CUDA& operator=(const CUDA&);
 protected:
  cudaStream_t stream_;
 public:
  explicit CUDA() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
  }
  virtual ~CUDA() {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }

  inline cudaStream_t stream() {
    return stream_;
  }
};

/**
 * @fn CUDA& cuda()
 * @brief cuda return a thread_local and static instance of CUDA.
 *        which contains thread specific cuda handles.
 */
CUDA& cuda();

/**
 * @fn cudaStream_t stream()
 * @brief stream returns a thread specific cudaStream_t handle to a cuda stream.
 *
 * the handle will come into being at the first call of stream() in the thread
 * and be destroyed when the thread exits.
 */
cudaStream_t stream();

}  // namespace tensor
