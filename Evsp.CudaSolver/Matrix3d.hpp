#pragma once

#include <assert.h>
#include <stdexcept>

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CudaCheck.h"


template<typename T>
class Matrix3d
{
public:
	Matrix3d(size_t sizeX, size_t sizeY, size_t sizeZ);
	Matrix3d(size_t sizeX, size_t sizeY, size_t sizeZ, T initValue);
	Matrix3d(Matrix3d<T> &other);
	~Matrix3d();

	Matrix3d<T>& operator=(const Matrix3d<T> &rhs) { assert(false); return *this; }

	/// <summary>
	/// Liefert einen Zeiger auf eine Kopie der Liste im Device-Speicher. Beim ersten Aufruf wird die Kopie erstellt.
	/// Alle nachfolgenden Aufrufe liefern einen Zeiger auf diese Kopie.
	/// Die in der Liste verwaltete Klasse muss ebenfalls eine getDevPtr-Methode bereitstellen.
	/// </summary>
	/// <returns>Zeiger auf die Kopie der Liste im Device-Speicher</returns>
	Matrix3d<T>* getDevPtr();

	/// <summary>
	/// Synchronisiert das Objekt mit der im Device-Speicher vorliegenden Kopie. 
	/// Die Daten werden also aus dem Device-Speicher in den Host-Speicher übertragen.
	/// </summary>
	void copyToHost();

	/// <summary>
	/// Synchronisiert die im Device-Speicher vorliegende Kopie mit dem Objekt. 
	/// Die Daten werden also aus dem Host-Speicher in den Device-Speicher übertragen.
	/// </summary>
	void copyToDevice();

	CU_HSTDEV size_t getSizeX() { return _sizeX; }
	CU_HSTDEV size_t getSizeY() { return _sizeY; }
	CU_HSTDEV size_t getSizeZ() { return _sizeZ; }

	CU_HSTDEV void set(size_t x, size_t y, size_t z, T value);
	CU_HSTDEV T get(size_t x, size_t y, size_t z);
	CU_HSTDEV T& itemAt(size_t x, size_t y, size_t z);

	void setAll(T value);

protected:
	Matrix3d();
	CU_HSTDEV size_t index(size_t x, size_t y, size_t z);
	size_t _sizeX;
	size_t _sizeY;
	size_t _sizeZ;
	T* _data;
	Matrix3d<T> *_devicePtr;
};


template<typename T>
size_t Matrix3d<T>::index(size_t x, size_t y, size_t z)
{
	size_t retVal = x + y * _sizeX + z * _sizeX * _sizeY;
	assert(retVal < _sizeX * _sizeY * _sizeZ);
	return retVal;
}


template<typename T>
Matrix3d<T>::Matrix3d()
	: _sizeX(0), _sizeY(0), _sizeZ(0), _data(0)
#ifdef __CUDACC__
	, _devicePtr(0)
#endif
{}

template<typename T>
Matrix3d<T>::Matrix3d(size_t sizeX, size_t sizeY, size_t sizeZ)
	: _sizeX(sizeX), _sizeY(sizeY), _sizeZ(sizeZ), _data(0)
#ifdef __CUDACC__
	, _devicePtr(0)
#endif
{
	size_t size = sizeX * sizeY * sizeZ;
	if (size <= 0) throw std::invalid_argument("Eines der Argumente ist negativ oder gleich Null!");
	_data = new T[size];
}


template<typename T>
Matrix3d<T>::Matrix3d(size_t sizeX, size_t sizeY, size_t sizeZ, T initValue)
	: Matrix3d<T>(sizeX, sizeY, sizeZ)
{
	setAll(initValue);
}


template<typename T>
Matrix3d<T>::Matrix3d(Matrix3d<T> &other)
	: Matrix3d<T>(other._sizeX, other._sizeY, other._sizeZ)
{
	for (int x = 0; x < _sizeX; x++) {
		for (int y = 0; y < _sizeY; y++) {
			for (int z = 0; z < _sizeZ; z++) {
				set(x, y, z, other.get(x, y, z));
			}
		}
	}
}


template<typename T>
Matrix3d<T>::~Matrix3d()
{
	if (_data) delete[] _data;
#ifdef __CUDACC__
	if (_devicePtr) {
		Matrix3d<T> *tempDevPtr = _devicePtr;
		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(Matrix3d<T>), cudaMemcpyDeviceToHost));
		if (_data) {
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _data));
		}
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, tempDevPtr));
	}
#endif
}


template<typename T>
Matrix3d<T>* Matrix3d<T>::getDevPtr()
{
#ifdef __CUDACC__
	if (!_devicePtr) {

		Matrix3d<T> *tempPtr;
		T *tempData = _data;

		size_t size = sizeof(T)*_sizeX*_sizeY*_sizeZ;
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_data, size));
		CUDA_CHECK(cudaMemcpy(_data, tempData, size, cudaMemcpyHostToDevice));

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempPtr, sizeof(Matrix3d<T>)));
		CUDA_CHECK(cudaMemcpy(tempPtr, this, sizeof(Matrix3d<T>), cudaMemcpyHostToDevice));

		_devicePtr = tempPtr;
		_data = tempData;
	}
#endif

	return _devicePtr;
}


template<typename T>
void Matrix3d<T>::copyToHost()
{
	if (_devicePtr) {
#ifdef __CUDACC__
		Matrix3d<T> *tempPtr = _devicePtr;
		T* tempData = _data;

		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(Matrix3d<T>), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(tempData, _data, sizeof(T) * _sizeX * _sizeY * _sizeZ, cudaMemcpyDeviceToHost));

		_data = tempData;
		_devicePtr = tempPtr;
#endif
	}
}


template<typename T>
void Matrix3d<T>::copyToDevice()
{
	if (_devicePtr) {
#ifdef __CUDACC__
		Matrix3d<T> *tempPtr = _devicePtr;
		T* tempData = _data;

		_devicePtr = 0;
		CUDA_CHECK(cudaMemcpy(this, tempPtr, sizeof(Matrix3d<T>), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(_data, tempData, sizeof(T)*_sizeX*_sizeY*_sizeZ, cudaMemcpyHostToDevice));

		_data = tempData;
		_devicePtr = tempPtr;
#endif
	}
}

template<typename T>
CU_HSTDEV void Matrix3d<T>::set(size_t x, size_t y, size_t z, T value)
{
	assert(x < _sizeX);
	assert(y < _sizeY);
	assert(z < _sizeZ);
	if (x < _sizeX && y < _sizeY && z < _sizeZ) {
		_data[index(x,y,z)] = value;
	}
}


template<typename T>
CU_HSTDEV T Matrix3d<T>::get(size_t x, size_t y, size_t z)
{
	assert(x < _sizeX);
	assert(y < _sizeY);
	assert(z < _sizeZ);
	return _data[index(x, y, z)];
}


template<typename T>
CU_HSTDEV T& Matrix3d<T>::itemAt(size_t x, size_t y, size_t z)
{
	assert(x < _sizeX);
	assert(y < _sizeY);
	assert(z < _sizeZ);
	return _data[index(x, y, z)];
}


template<typename T>
void Matrix3d<T>::setAll(T value)
{
	size_t arraySize = _sizeX * _sizeY * _sizeZ;
	for (size_t i = 0; i < arraySize; i++) _data[i] = value;
}

