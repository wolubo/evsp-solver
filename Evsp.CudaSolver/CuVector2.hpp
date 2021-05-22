#pragma once

#include <stdexcept>
#include <assert.h>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CudaCheck.h"






/// <summary>
/// Stellt einen Vektor fixer (also zur Compile-Zeit bekannter Größe) bereit.
/// </summary>
template<int SIZE>
class CuVector2
{
public:
	CuVector2();
	CuVector2(const CuVector2 &other) { assign(false); }
	explicit CuVector2(int initValue);
	~CuVector2();

	CuVector2& operator=(const CuVector2 &rhs) { assert(false); return *this; }

	/// <summary>
	/// Liefert einen Zeiger auf eine Kopie des Objekts im Device-Speicher. Beim ersten Aufruf wird die Kopie erstellt.
	/// Alle nachfolgenden Aufrufe liefern einen Zeiger auf diese Kopie.
	/// Die in der Liste verwaltete Klasse muss ebenfalls eine getDevPtr-Methode bereitstellen.
	/// </summary>
	/// <returns>Zeiger auf die Kopie im Device-Speicher</returns>
	CuVector2<SIZE>* getDevPtr();

	/// <summary>
	/// Synchronisiert das Objekt mit der im Device-Speicher vorliegenden Kopie. 
	/// Die Daten werden also aus dem Device-Speicher in den Host-Speicher übertragen.
	/// </summary>
	void copyToHost();

	/// <summary>
	/// Synchronisiert die im Device-Speicher vorliegende Kopie mit dem Objekt. 
	/// Die Daten werden also aus dem Host-Speicher in den Device-Speicher übertragen.
	/// </summary>
	void sync2device();

	CU_HSTDEV int& operator[](int pos);

	CU_HSTDEV bool operator==(CuVector2<SIZE> &other);
	CU_HSTDEV bool operator!=(CuVector2<SIZE> &other);

	CU_HSTDEV void setAll(int value);

protected:
	int _data[SIZE];
	CuVector2<SIZE> *_devicePtr;
};


template<int SIZE>
CuVector2<SIZE>::CuVector2()
	: _data(), _devicePtr(0)
{
}


template<int SIZE>
CuVector2<SIZE>::CuVector2(int initValue)
	: CuVector2<SIZE>()
{
	setAll(initValue);
}


template<int SIZE>
CuVector2<SIZE>::~CuVector2()
{
#ifdef __CUDACC__
	if (_devicePtr) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
#endif
}


template<int SIZE>
CuVector2<SIZE>* CuVector2<SIZE>::getDevPtr()
{
#ifdef __CUDACC__
	if (!_devicePtr) {
		CuVector2<SIZE>* temp;
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuVector2<SIZE>)));
		
		CUDA_CHECK(cudaMemcpy(temp, this, sizeof(CuVector2<SIZE>), cudaMemcpyHostToDevice));
		_devicePtr = temp;
	}
#endif
	return _devicePtr;
}


template<int SIZE>
void CuVector2<SIZE>::copyToHost()
{
#ifdef __CUDACC__
	if (_devicePtr) {
		CuVector2<SIZE>* temp = _devicePtr;
		CUDA_CHECK(cudaMemcpy(&temp, _devicePtr, sizeof(CuVector2<SIZE>), cudaMemcpyDeviceToHost));
		_devicePtr = temp;
	}
#endif
}



template<int SIZE>
void CuVector2<SIZE>::sync2device()
{
#ifdef __CUDACC__
	assert(false);

	CuVector2<SIZE>* retVal = 0;
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(retVal), sizeof(CuVector2<SIZE>)));
	

	int* dataOnHost = _data;
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(_data), sizeof(int) * _size));
	
	CUDA_CHECK(cudaMemcpy(_data, dataOnHost, sizeof(int) * _size, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(retVal, this, sizeof(CuVector2<SIZE>), cudaMemcpyHostToDevice));
	_data = dataOnHost;
#endif
}


template<int SIZE>
CU_HSTDEV int& CuVector2<SIZE>::operator[](int pos)
{
	assert(pos >= 0);
	assert(pos < SIZE);
	return _data[pos];
}


template<int SIZE>
CU_HSTDEV bool CuVector2<SIZE>::operator==(CuVector2<SIZE> &other)
{
	for (int i = 0; i < SIZE; i++) {
		int left = _data[i];
		int right = other._data[i];
		if (left != right) {
			return false;
		}
	}
	return true;
}


template<int SIZE>
CU_HSTDEV bool CuVector2<SIZE>::operator!=(CuVector2<SIZE> &other)
{
	return !(this->operator==(other));
}


template<int SIZE>
CU_HSTDEV void CuVector2<SIZE>::setAll(int value)
{
	for (int i = 0; i < SIZE; i++) {
		_data[i] = value;
	}
}

