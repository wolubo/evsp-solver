#include "Managed.h"

#include "cuda_runtime.h"
#include "CudaCheck.h"





void * Managed::operator new(size_t len) {
	void *ptr;
	CUDA_CHECK(cudaMallocManaged(&ptr, len));
	CUDA_CHECK(cudaDeviceSynchronize());
	return ptr;
}

void Managed::operator delete(void *ptr) {
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, ptr));
}

void* Managed::operator new[](size_t len)
{
	void *ptr;
	CUDA_CHECK(cudaMallocManaged(&ptr, len));
	CUDA_CHECK(cudaDeviceSynchronize());
	return ptr;
}

void Managed::operator delete[](void *ptr)
{
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, ptr));
}

