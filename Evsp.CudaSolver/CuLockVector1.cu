#include "CuLockVector1.h"
#include "cuda_runtime.h"
#include "CudaCheck.h"



CuLockVector1::CuLockVector1(int size)
	: _devPtr(0)
{
	_size = size;
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_lock, _size * sizeof(int)));
	CUDA_CHECK(cudaMemset(_lock, 0, _size * sizeof(int)));
}


CuLockVector1::~CuLockVector1()
{
	if (_lock) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _lock));
	if (_devPtr) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devPtr));
}


CuLockVector1* CuLockVector1::getDevPtr()
{
	if (!_devPtr) {
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devPtr, sizeof(CuLockVector1)));
		CUDA_CHECK(cudaMemcpy(_devPtr, this, sizeof(CuLockVector1), cudaMemcpyHostToDevice));
	}
	return _devPtr;
}


__device__ bool CuLockVector1::lock(int index)
{
	if (index >= _size) return false;
	int old;
	old = atomicCAS(&(_lock[index]), 0, 1);
	return (old == 0);
}


__device__ void CuLockVector1::unlock(int index)
{
	_lock[index] = 0;
}
