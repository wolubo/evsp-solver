#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "CudaCheck.h"


///<summary>
/// Realisiert einen Lock-Mechanismus für Vektoren fixer Größe. 
///</summary>
template<unsigned int SIZE>
class CuLockVector2
{
public:
	CuLockVector2();
	~CuLockVector2();
	CuLockVector2<SIZE>* getDevPtr();
	CU_DEV bool lock(int index);
	CU_DEV void unlock(int index);
private:
	int _lock[SIZE];
	CuLockVector2<SIZE> *_deviceObject;
};


template<unsigned int SIZE>
CuLockVector2<SIZE>::CuLockVector2()
	: _deviceObject(0)
{
	for (unsigned int i = 0; i < SIZE; i++) {
		_lock[i] = 0;
	}
}


template<unsigned int SIZE>
CuLockVector2<SIZE>::~CuLockVector2()
{
	if (_deviceObject) {
#ifdef __CUDACC__ 
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _deviceObject));
#endif
	}
}


template<unsigned int SIZE>
CuLockVector2<SIZE>* CuLockVector2<SIZE>::getDevPtr()
{
#ifdef __CUDACC__ 
	if (!_deviceObject) {
		CuLockVector2<SIZE>* tempDevObj;
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempDevObj, sizeof(CuLockVector2<SIZE>)));
		
		CUDA_CHECK(cudaMemcpy(tempDevObj, this, sizeof(CuLockVector2<SIZE>), cudaMemcpyHostToDevice));
		_deviceObject = tempDevObj;
	}
#endif
	return _deviceObject;
}


template<unsigned int SIZE>
CU_DEV bool CuLockVector2<SIZE>::lock(int index)
{
	if (index >= SIZE) {
		return false;
	}
	int old = 1;
#ifdef __CUDACC__ 
	old = atomicCAS(&(_lock[index]), 0, 1);
#endif
	return (old == 0);
}


template<unsigned int SIZE>
CU_DEV void CuLockVector2<SIZE>::unlock(int index)
{
	if (index >= SIZE) return;
#ifdef __CUDACC__ 
	atomicExch(&(_lock[index]), 0);
#endif
}

