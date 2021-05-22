#pragma once

#ifndef __CUDACC__
#include <atomic>
#endif

#include <assert.h>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"


/// <summary>
/// Verwaltet Bit-Werte in einem Vektor variabler Größe. 
/// Optimiert für einen möglichst geringen Speicherbedarf.
/// Nicht für konkurierende Zugriffe geeignet (nicht thread-safe)!
/// </summary>
class CuBoolVector1
{
public:
	explicit CuBoolVector1(int size, bool initialValue = false);

#pragma warning(once:26495)
	CuBoolVector1(const CuBoolVector1 &other) { assert(false); }

	~CuBoolVector1();
	CuBoolVector1& operator=(const CuBoolVector1 &rhs) { assert(false); return *this; }
	CuBoolVector1* getDevPtr();
	void copyToHost();
	void copyToDevice();
	CU_DEV void setAll();
	CU_DEV void unsetAll();
	CU_DEV void set(int index);
	CU_DEV void unset(int index);
	CU_DEV void toggle(int index);
	CU_DEV bool get(int index);
private:
#ifdef WIN64
	typedef unsigned long long base_type;
	const base_type ONE = 1ull;
	const base_type ZERO = 0ull;
#else
	typedef unsigned int base_type;
	const base_type ONE = 1u;
	const base_type ZERO = 0u;
#endif
#ifdef __CUDACC__
	typedef base_type stored_base_type;
#else
	typedef std::atomic<base_type> stored_base_type;
#endif
	CU_DEV void setThreadSafe(int arrayIndex, base_type newValue);
	int _bitSize;
	int _arraySize;
	stored_base_type *_data;
	CuBoolVector1 *_devicePtr;
};


inline CuBoolVector1::CuBoolVector1(int size, bool initialValue)
	: _bitSize(size), _arraySize((size + sizeof(base_type) - 1) / sizeof(base_type))
{
	assert(sizeof(base_type) == sizeof(stored_base_type)); // Andernfalls muss die Implementierung komplett überarbeitet werden!
	_devicePtr = 0;
	_data = new stored_base_type[_arraySize];
	if (initialValue) {
		memset(_data, -1, sizeof(stored_base_type)*_arraySize);
	}
	else {
		memset(_data, 0, sizeof(stored_base_type)*_arraySize);
	}
}


inline CuBoolVector1::~CuBoolVector1()
{
	if (_data) delete[] _data;
	if (_devicePtr) {
#ifdef __CUDACC__
		CuBoolVector1 *tempDevPtr = _devicePtr;
		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuBoolVector1), cudaMemcpyDeviceToHost));
		if (_data) {
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _data));
		}
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, tempDevPtr));
#endif
	}
}


inline CuBoolVector1* CuBoolVector1::getDevPtr()
{
	if (!_devicePtr) {
#ifdef __CUDACC__
		CuBoolVector1 *tempObj = 0;
		base_type *tempData = _data;
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_data, sizeof(stored_base_type)*_arraySize));
		
		CUDA_CHECK(cudaMemcpy(_data, tempData, sizeof(stored_base_type)*_arraySize, cudaMemcpyHostToDevice));
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempObj, sizeof(CuBoolVector1)));
		
		CUDA_CHECK(cudaMemcpy(tempObj, this, sizeof(CuBoolVector1), cudaMemcpyHostToDevice));
		_devicePtr = tempObj;
		_data = tempData;
#endif
	}
	return _devicePtr;
}


inline void CuBoolVector1::copyToHost()
{
#ifdef __CUDACC__
	if (_devicePtr) {
		CuBoolVector1* tempDevObj = _devicePtr;
		base_type* tempData = _data;
		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuBoolVector1), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(tempData, _data, sizeof(stored_base_type) * _arraySize, cudaMemcpyDeviceToHost));
		_devicePtr = tempDevObj;
		_data = tempData;
	}
#endif
}


inline void CuBoolVector1::copyToDevice()
{
	if (_devicePtr) {
#ifdef __CUDACC__
		CuBoolVector1 *tempObj = _devicePtr;
		base_type* tempData = _data;

		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuBoolVector1), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(_data, tempData, sizeof(stored_base_type)*_arraySize, cudaMemcpyHostToDevice));

		_data = tempData;
		_devicePtr = tempObj;
#endif
	}
}

inline CU_DEV void CuBoolVector1::setAll()
{
	for (int i = 0; i < _arraySize; i++) {
		setThreadSafe(i, ~ZERO);
	}
}


inline CU_DEV void CuBoolVector1::unsetAll()
{
	for (int i = 0; i < _arraySize; i++) {
		setThreadSafe(i, ZERO);
	}
}


inline CU_DEV void CuBoolVector1::set(int index)
{
	int arrayIndex = index / sizeof(base_type);
	int bitIndex = index % sizeof(base_type);
	assert(arrayIndex < _arraySize);
	setThreadSafe(arrayIndex, _data[arrayIndex] | (ONE << bitIndex)); // or
}


inline CU_DEV void CuBoolVector1::unset(int index)
{
	int arrayIndex = index / sizeof(base_type);
	int bitIndex = index % sizeof(base_type);
	assert(arrayIndex < _arraySize);
	setThreadSafe(arrayIndex, _data[arrayIndex] & ~(ONE << bitIndex)); // and
}


inline CU_DEV void CuBoolVector1::toggle(int index)
{
	int arrayIndex = index / sizeof(base_type);
	int bitIndex = index % sizeof(base_type);
	assert(arrayIndex < _arraySize);
	setThreadSafe(arrayIndex, _data[arrayIndex] ^ (ONE << bitIndex)); // and
}


inline CU_DEV bool CuBoolVector1::get(int index)
{
	int arrayIndex = index / sizeof(base_type);
	int bitIndex = index % sizeof(base_type);
	assert(arrayIndex < _arraySize);
	base_type value = _data[arrayIndex] & (ONE << bitIndex);
	return value != 0;
}


inline CU_DEV void CuBoolVector1::setThreadSafe(int arrayIndex, base_type newValue)
{
	base_type oldValue;

#ifdef __CUDACC__
	base_type result;
	oldValue = _data[arrayIndex];
	do {
		result = atomicCAS(&(_data[arrayIndex]), oldValue, newValue);
	} while (result != oldValue);
#else
	bool exchanged;
	do {
		oldValue = _data[arrayIndex];
		exchanged = _data[arrayIndex].compare_exchange_strong(oldValue, newValue);
	} while (!exchanged);
#endif
}
