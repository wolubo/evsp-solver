#pragma once

#include <stdexcept>
#include <assert.h>
#include <typeinfo>
#include <string>

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CudaCheck.h"

///<summary>
/// Realisiert eine Matrix variabler (also zur Compile-Zeit unbekannter) Größe. 
///</summary>
template<typename T>
class CuMatrix1
{
public:
	CuMatrix1() = delete;
	CuMatrix1(int numOfRows, int numOfCols);
	CuMatrix1(const CuMatrix1<T> &other);
	~CuMatrix1();

	CuMatrix1<T>& operator=(const CuMatrix1<T> &rhs);

	/// <summary>
	/// Setzt den Inhalt auf '0', indem der Speicherbereich mit Nullen überschrieben wird.
	/// </summary>
	CU_HSTDEV void initialize();

	/// <summary>
	/// Liefert einen Zeiger auf eine Kopie der Liste im Device-Speicher. Beim ersten Aufruf wird die Kopie erstellt.
	/// Alle nachfolgenden Aufrufe liefern einen Zeiger auf diese Kopie.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Device-Speicher</returns>
	CuMatrix1<T>* getDevPtr();

	/// <summary>
	/// Synchronisiert das Objekt mit der im Device-Speicher vorliegenden Kopie. 
	/// Die Daten werden also aus dem Device-Speicher in den Host-Speicher übertragen. 
	/// </summary>
	void copyToHost();

	/// <summary>
	/// Überschreibt die Daten im Device-Speicher mit denen des Objekts. Kopiert Änderungen vom Host also aufs Device.
	/// Das Objekt muss vorab bereits mit getDevPtr() in den Device-Speicher übertragen worden sein.
	/// </summary>			
	void copyToDevice();

	/// <summary>
	/// Synchronisiert das Objekt mit der im Device-Speicher vorliegenden Kopie. 
	/// Die Daten werden also aus dem Device-Speicher in den Host-Speicher übertragen. Dabei wird für jedes einzelne Element
	/// die Fuktion 'translate(T)' aufgerufen. Sollte es sich bei den einzelnen Elementen um Pointer handeln können
	/// die Objekte, auf welche die Pointer verweisen auf diese Weise ebenfalls kopiert werden!
	/// </summary>
	//void copyToHost(T translate(T));

	CU_HSTDEV void set(int row, int col, T value);
	CU_HSTDEV T get(int row, int col) const;
	CU_HSTDEV T& itemAt(int row, int col) const;

	CU_HSTDEV void setAll(T value);
	void setAllOnDevice(T value);

	CU_HSTDEV bool operator==(CuMatrix1 &other);
	CU_HSTDEV bool operator!=(CuMatrix1 &other);

	CU_HSTDEV int getNumOfRows() { return _numOfRows; }
	CU_HSTDEV int getNumOfCols() { return _numOfCols; }

protected:
	int _numOfRows;
	int _numOfCols;
	T* _data;
	CuMatrix1<T> *_devicePtr;
};


template<typename T>
CuMatrix1<T>::CuMatrix1(int numOfRows, int numOfCols)
	: _data(0), _devicePtr(0), _numOfRows(numOfRows), _numOfCols(numOfCols)
{
	assert(numOfRows > 0);
	assert(numOfCols > 0);
	_data = new T[numOfRows * numOfCols];
	//_data = (T*) malloc(sizeof(T) * numOfRows * numOfCols);
	initialize();
}


template<typename T>
CuMatrix1<T>::CuMatrix1(const CuMatrix1<T> &other)
	: CuMatrix1<T>()
{
	assert(false);
}


template<typename T>
CuMatrix1<T>::~CuMatrix1()
{
	if (_data) {
		delete[] _data;
	}
#ifdef __CUDACC__
	if (_devicePtr) {
		CuMatrix1<T> *tempDevPtr = _devicePtr;
		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuMatrix1<T>), cudaMemcpyDeviceToHost));
		if (_data) {
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _data));
		}
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, tempDevPtr));
	}
#endif
}


template<typename T>
CuMatrix1<T>& CuMatrix1<T>::operator=(const CuMatrix1<T> &rhs)
{
	if (this != &rhs) {
		if (_data) {
			delete[] _data;
			_data = 0;
		}
		if (_devicePtr) {
#ifdef __CUDACC__
			CuMatrix1<T> *tempDevPtr = _devicePtr;
			CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuMatrix1<T>), cudaMemcpyDeviceToHost));
			if (_data) {
				CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _data));
			}
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, tempDevPtr));
#endif
			_devicePtr = 0;
		}

		_numOfRows = rhs._numOfRows;
		_numOfCols = rhs._numOfCols;

		if (rhs._data) {
			_data = new T[_numOfRows * _numOfCols];
		}

		for (int r = 0; r < _numOfRows; r++) {
			for (int c = 0; c < _numOfCols; c++) {
				set(r, c, rhs.get(r, c));
			}
		}
	}

	return *this;
}


template<typename T>
void CuMatrix1<T>::initialize()
{
	memset(_data, 0, sizeof(T) * _numOfRows * _numOfCols);
}


template<typename T>
CuMatrix1<T>* CuMatrix1<T>::getDevPtr()
{
#ifdef __CUDACC__
	if (!_devicePtr) {
		T* tempHostData = _data;
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_data, sizeof(T)*_numOfRows*_numOfCols));

		CUDA_CHECK(cudaMemcpy(_data, tempHostData, sizeof(T)*_numOfRows*_numOfCols, cudaMemcpyHostToDevice));

		CuMatrix1<T>* tempDevObj;
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempDevObj, sizeof(CuMatrix1<T>)));

		CUDA_CHECK(cudaMemcpy(tempDevObj, this, sizeof(CuMatrix1<T>), cudaMemcpyHostToDevice));
		_data = tempHostData;
		_devicePtr = tempDevObj;
	}
#endif
	return _devicePtr;
}


template<typename T>
void CuMatrix1<T>::copyToHost()
{
	if (_devicePtr) {
#ifdef __CUDACC__
		if (!_data) _data = new T[_numOfRows * _numOfCols];
		CuMatrix1<T> *temp = _devicePtr;
		T* data = _data;
		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuMatrix1<T>), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(data, _data, sizeof(T)*_numOfRows*_numOfCols, cudaMemcpyDeviceToHost));
		_devicePtr = temp;
		_data = data;
#endif
	}
}


template<typename T>
void CuMatrix1<T>::copyToDevice()
{
	if (_devicePtr) {
#ifdef __CUDACC__
		CuMatrix1<T> *tempObj = _devicePtr;
		T* tempData = _data;

		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuMatrix1<T>), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(_data, tempData, sizeof(T)*_numOfRows*_numOfCols, cudaMemcpyHostToDevice));

		_data = tempData;
		_devicePtr = tempObj;
#endif
	}
}


template<typename T>
CU_HSTDEV void CuMatrix1<T>::set(int row, int col, T value)
{
	assert(row >= 0);
	assert(col >= 0);
	assert(row < _numOfRows);
	assert(col < _numOfCols);
	_data[row * _numOfCols + col] = value;
}


template<typename T>
CU_HSTDEV T CuMatrix1<T>::get(int row, int col) const
{
	assert(row >= 0);
	assert(col >= 0);
	assert(row < _numOfRows);
	assert(col < _numOfCols);
	return _data[row * _numOfCols + col];
}


template<typename T>
CU_HSTDEV T& CuMatrix1<T>::itemAt(int row, int col) const
{
	assert(row >= 0);
	assert(col >= 0);
	assert(row < _numOfRows);
	assert(col < _numOfCols);
	return _data[row * _numOfCols + col];
}


template<typename T>
CU_HSTDEV bool CuMatrix1<T>::operator==(CuMatrix1<T> &other)
{
	int arraySize = _numOfRows * _numOfCols;
	for (int i = 0; i < arraySize; i++) {
		T left = _data[i];
		T right = other._data[i];
		if (left != right) {
			return false;
		}
	}
	return true;
}


template<typename T>
CU_HSTDEV bool CuMatrix1<T>::operator!=(CuMatrix1<T> &other)
{
	return !(this->operator==(other));
}


template<typename T>
CU_HSTDEV void CuMatrix1<T>::setAll(T value)
{
	int arraySize = _numOfRows * _numOfCols;
	for (int i = 0; i < arraySize; i++) _data[i] = value;
}


template<typename T>
CU_GLOBAL void setAllKernel(CuMatrix1<T> *retVal, int numOfRows, int numOfCols, T initValue)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < numOfRows && col < numOfCols) {
		retVal->set(row, col, initValue);
	}
}


template<typename T>
void CuMatrix1<T>::setAllOnDevice(T value)
{
#ifdef __CUDACC__
	if (getDevPtr()) {
		const int blockSize = 32;
		int rowBlocks = (_numOfRows + blockSize - 1) / blockSize;
		int colBlocks = (_numOfCols + blockSize - 1) / blockSize;

		dim3 dimGrid(rowBlocks, colBlocks);
		dim3 dimBlock(blockSize, blockSize);
		setAllKernel<T> << <dimGrid, dimBlock >> > (_devicePtr, _numOfRows, _numOfCols, value);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
#endif
}

