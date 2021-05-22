#include "CuLockMatrix1.h"
#include <assert.h>
#include "cuda_runtime.h"
#include "CudaCheck.h"



CuLockMatrix1::CuLockMatrix1(int numOfRows, int numOfCols)
: _numOfRows(numOfRows), _numOfCols(numOfCols), _devPtr(0)
{
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_lock, _numOfRows * _numOfCols * sizeof(int)));
	CUDA_CHECK(cudaMemset(_lock, 0, _numOfRows * _numOfCols * sizeof(int)));
}


CuLockMatrix1::~CuLockMatrix1()
{
	if (_lock) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _lock));
	if (_devPtr) CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devPtr));
}


CuLockMatrix1* CuLockMatrix1::getDevPtr()
{
	if (!_devPtr) {
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devPtr, sizeof(CuLockMatrix1)));
		CUDA_CHECK(cudaMemcpy(_devPtr, this, sizeof(CuLockMatrix1), cudaMemcpyHostToDevice));
	}
	return _devPtr;
}


CU_DEV bool CuLockMatrix1::lock(int row, int col)
{
	assert(row < _numOfRows);
	assert(col < _numOfCols);
	if (row  >= _numOfRows) return false;
	if (col >= _numOfCols) return false;
	int old;
	old = atomicCAS(&(_lock[row*_numOfCols+col]), 0, 1);
	return (old == 0);
}


CU_DEV void CuLockMatrix1::unlock(int row, int col)
{
	assert(row < _numOfRows);
	assert(col < _numOfCols);
	if (row >= _numOfRows) return;
	if (col >= _numOfCols) return;
	_lock[row*_numOfCols + col] = 0;
}
