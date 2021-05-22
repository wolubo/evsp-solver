#pragma once

#include <stdexcept>
#include <assert.h>
#include <typeinfo>
#include <string>

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CudaCheck.h"


///<summary>
/// Realisiert eine Matrix fixer (also zur Compile-Zeit bekannter) Größe. 
///</summary>
template<typename T, int ROWS, int COLS>
class CuMatrix2
{
public:
	CuMatrix2();
	CuMatrix2(const CuMatrix2<T, ROWS, COLS> &other);
	~CuMatrix2();

	CuMatrix2<T, ROWS, COLS>& operator=(const CuMatrix2<T, ROWS, COLS> &rhs) { assert(false); return *this; }

	/// <summary>
	/// Liefert einen Zeiger auf eine Kopie der Liste im Device-Speicher. Beim ersten Aufruf wird die Kopie erstellt.
	/// Alle nachfolgenden Aufrufe liefern einen Zeiger auf diese Kopie.
	/// Die in der Liste verwaltete Klasse muss ebenfalls eine getDevPtr-Methode bereitstellen.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Device-Speicher</returns>
	CuMatrix2<T, ROWS, COLS>* getDevPtr();
	//CuMatrix2<T,ROWS,COLS>* getDevPtr(T translate(T));

	/// <summary>
	/// Synchronisiert das Objekt mit der im Device-Speicher vorliegenden Kopie. 
	/// Die Daten werden also aus dem Device-Speicher in den Host-Speicher übertragen. Sollte es sich bei den einzelnen
	/// Elementen um Pointer handeln, so müssen die Objekte, auf welche die Pointer verweisen danach ebenfalls noch kopiert
	/// werden!
	/// </summary>
	void copyToHost();

	/// <summary>
	/// Synchronisiert das Objekt mit der im Device-Speicher vorliegenden Kopie. 
	/// Die Daten werden also aus dem Device-Speicher in den Host-Speicher übertragen. Dabei wird für jedes einzelne Element
	/// die Fuktion 'translate(T)' aufgerufen. Sollte es sich bei den einzelnen Elementen um Pointer handeln können
	/// die Objekte, auf welche die Pointer verweisen auf diese Weise ebenfalls kopiert werden!
	/// </summary>
	//void copy2host(T translate(T));

	/// <summary>
	/// Synchronisiert die im Device-Speicher vorliegende Kopie mit dem Objekt. 
	/// Die Daten werden also aus dem Host-Speicher in den Device-Speicher übertragen.
	/// </summary>
	void copy2device();

	CU_HSTDEV void set(int row, int col, T value);
	CU_HSTDEV T get(int row, int col);
	CU_HSTDEV T& itemAt(int row, int col);

	void setAll(T value);
	void setAllOnDevice(T value);

	bool operator==(CuMatrix2 &other);
	bool operator!=(CuMatrix2 &other);

	CU_HSTDEV int getNumOfRows() { return ROWS; }
	CU_HSTDEV int getNumOfCols() { return COLS; }

protected:
	T _data[ROWS*COLS];
#ifdef __CUDACC__
	CuMatrix2<T, ROWS, COLS> *_devicePtr;
#endif
};


template<typename T, int ROWS, int COLS>
CuMatrix2<T, ROWS, COLS>::CuMatrix2()
	: _data()
#ifdef __CUDACC__
	, _devicePtr(0)
#endif
{
}


template<typename T, int ROWS, int COLS>
CuMatrix2<T, ROWS, COLS>::CuMatrix2(const CuMatrix2<T, ROWS, COLS> &other)
	: CuMatrix2<T, ROWS, COLS>()
{
	assert(false);
}


template<typename T, int ROWS, int COLS>
CuMatrix2<T, ROWS, COLS>::~CuMatrix2()
{
#ifdef __CUDACC__
	if (_devicePtr) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
#endif
}


template<typename T, int ROWS, int COLS>
CuMatrix2<T, ROWS, COLS>* CuMatrix2<T, ROWS, COLS>::getDevPtr()
{
#ifdef __CUDACC__
	if (!_devicePtr) {
		CuMatrix2<T, ROWS, COLS>* temp;
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&temp, sizeof(CuMatrix2<T, ROWS, COLS>)));
		
		CUDA_CHECK(cudaMemcpy(temp, this, sizeof(CuMatrix2<T, ROWS, COLS>), cudaMemcpyHostToDevice));
		_devicePtr = temp;
	}
#endif
	return _devicePtr;
}


//template<typename T, int ROWS, int COLS>
//CuMatrix2<T,ROWS,COLS>* CuMatrix2<T,ROWS,COLS>::getDevPtr(T translate(T))
//{
//#ifdef __CUDACC__
//	T *temp = new T[_numOfRows * COLS];

//	for (int row = 0; row < _numOfRows; row++) {
//		for (int col = 0; col < COLS; col++) {
//			T translatedItem = translate(get(row, col));
//			temp[row * COLS + col] = translatedItem;
//		}
//	}

//	T *dataSaved = _data;
//	_data = temp;
//	CuMatrix2<T,ROWS,COLS> * retVal = getDevPtr();
//	_data = dataSaved;
//	delete[] temp;

//	return retVal;
//#endif
//}


template<typename T, int ROWS, int COLS>
void CuMatrix2<T, ROWS, COLS>::copyToHost()
{
#ifdef __CUDACC__
	if (_devicePtr == 0) return;
	CuMatrix2<T, ROWS, COLS> *temp = _devicePtr;
	CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuMatrix2<T, ROWS, COLS>), cudaMemcpyDeviceToHost));
	_devicePtr = temp;
#endif
}


//#ifdef __CUDACC__
//			template<typename T, int ROWS, int COLS>
//			void CuMatrix2<T,ROWS,COLS>::copy2host(T translate(T))
//			{
//				copy2host();
//				if (_data) {
//					for (int row = 0; row < _numOfRows; row++) {
//						for (int col = 0; col < COLS; col++) {
//							T translatedItem = translate(get(row, col));
//							set(row, col, translatedItem);
//						}
//					}
//				}
//			}
//#endif


//#ifdef __CUDACC__
//			template<typename T, int ROWS, int COLS>
//			void CuMatrix2<T,ROWS,COLS>::copy2device()
//			{
//				assert(false);
//
//				if (!hostPtr || !hostPtr->_data) return;
//
//				CuMatrix2<T,ROWS,COLS>* retVal = 0;
//				CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(retVal), sizeof(CuMatrix2<T,ROWS,COLS>)));
//				if (!retVal) return;
//
//				int size = hostPtr->_numOfRows*hostPtr->COLS;
//
//				T* dataOnDevice;
//				CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(dataOnDevice), sizeof(T) * size));
//				if (!dataOnDevice) {
//					CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, retVal));
//					return;
//				}
//
//				CUDA_CHECK(cudaMemcpy(dataOnDevice, hostPtr->_data, sizeof(T) * size, cudaMemcpyHostToDevice));
//
//				CuMatrix2 temp;
//				temp._numOfRows = hostPtr->_numOfRows;
//				temp.COLS = hostPtr->COLS;
//				temp._data = dataOnDevice;
//				CUDA_CHECK(cudaMemcpy(retVal, &temp, sizeof(CuMatrix2<T,ROWS,COLS>), cudaMemcpyHostToDevice));
//			}
//#endif


template<typename T, int ROWS, int COLS>
CU_HSTDEV void CuMatrix2<T, ROWS, COLS>::set(int row, int col, T value)
{
	assert(row >= 0);
	assert(col >= 0);
	assert(row < ROWS);
	assert(col < COLS);
	if (row < ROWS && col < COLS) {
		_data[row * COLS + col] = value;
	}
}


template<typename T, int ROWS, int COLS>
T CuMatrix2<T, ROWS, COLS>::get(int row, int col)
{
	assert(row >= 0);
	assert(col >= 0);
	assert(row < ROWS);
	assert(col < COLS);
	return _data[row*COLS + col];
}


template<typename T, int ROWS, int COLS>
CU_HSTDEV T& CuMatrix2<T, ROWS, COLS>::itemAt(int row, int col)
{
	assert(row >= 0);
	assert(col >= 0);
	assert(row < ROWS);
	assert(col < COLS);
	return _data[row*COLS + col];
}


template<typename T, int ROWS, int COLS>
bool CuMatrix2<T, ROWS, COLS>::operator==(CuMatrix2<T, ROWS, COLS> &other)
{
	int arraySize = ROWS * COLS;
	for (int i = 0; i < arraySize; i++) {
		T left = _data[i];
		T right = other._data[i];
		if (left != right) {
			return false;
		}
	}
	return true;
}


template<typename T, int ROWS, int COLS>
bool CuMatrix2<T, ROWS, COLS>::operator!=(CuMatrix2<T, ROWS, COLS> &other)
{
	return !(this->operator==(other));
}


template<typename T, int ROWS, int COLS>
void CuMatrix2<T, ROWS, COLS>::setAll(T value)
{
	int arraySize = ROWS*COLS;
	for (int i = 0; i < arraySize; i++) _data[i] = value;
}


template<typename T, int ROWS, int COLS>
CU_GLOBAL void setAllKernel(CuMatrix2<T, ROWS, COLS> *retVal, T initValue)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < ROWS && col < COLS) {
		retVal->set(row, col, initValue);
	}
}


template<typename T, int ROWS, int COLS>
void CuMatrix2<T, ROWS, COLS>::setAllOnDevice(T value)
{
#ifdef __CUDACC__
	if (getDevPtr()) {
		const int blockSize = 32;
		int rowBlocks = (ROWS + blockSize - 1) / blockSize;
		int colBlocks = (COLS + blockSize - 1) / blockSize;

		dim3 dimGrid(rowBlocks, colBlocks);
		dim3 dimBlock(blockSize, blockSize);
		setAllKernel<T> << <dimGrid, dimBlock >> > (_devicePtr, value);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
#endif
}

