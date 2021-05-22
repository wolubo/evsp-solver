#include "CudaCheck.h"

#include <stdio.h>
#include "cuda_runtime.h"


cudaError_t wbCudaMalloc(const char *file, int line, void **devPtr, size_t size)
{
	cudaError_t code = cudaMalloc(devPtr, size);

#ifdef WB_MEMCHECK
	printf("cudaMalloc: %p, %u bytes allocated (%s %i)\n", *devPtr, size, file, line);
#endif

	return code;
}


cudaError_t wbCudaFree(const char *file, int line, void *devPtr)
{
	cudaError_t code = cudaFree(devPtr);

#ifdef WB_MEMCHECK
	printf("cudaFree: %p (%s %d)\n", devPtr, file, line);
#endif

	return code;
}



#ifdef _DEBUG

void printMemStat(const char *file, int line)
{
	size_t free, total;
	float ffree, ftotal;
	CUDA_CHECK(cudaMemGetInfo(&free, &total));
	ffree = free / (1024.0f * 1024.0f);
	ftotal = total / (1024.0f * 1024.0f);
	printf("Memstat: %.0f mb of %.0f mb free (%s %d)\n", ffree, ftotal, file, line);
}
#endif

__host__ void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) {
		printf("GPUassert: %s (Code: %d) %s %d\n", cudaGetErrorString(code), code, file, line);
		cudaDeviceReset();
		exit(code);
	}
}

__device__ void gpuAssertDev(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) {
		printf("GPUassert: %i %s %d\n", code, file, line);
	}
}


