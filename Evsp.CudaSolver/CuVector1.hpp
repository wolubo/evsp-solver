#pragma once

#include <stdexcept>
#include <assert.h>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "CudaCheck.h"


/// <summary>
/// Stellt einen Vektor variabler (also zur Compile-Zeit unbekannter Größe) bereit.
/// </summary>
template<typename T>
class CuVector1
{
public:
	CuVector1() = delete;
	explicit CuVector1(int size);
	//CuVector1(int size, T initValue);
	CuVector1(const CuVector1<T>& other);
	~CuVector1();

	CuVector1<T>& operator=(const CuVector1<T> &rhs);

	/// <summary>
	/// Setzt den Inhalt auf '0', indem der Speicherbereich mit Nullen überschrieben wird.
	/// </summary>
	CU_HSTDEV void initialize();

	/// <summary>
	/// Liefert einen Zeiger auf eine Kopie des Objekts im Device-Speicher. Beim ersten Aufruf wird die Kopie erstellt.
	/// Alle nachfolgenden Aufrufe liefern einen Zeiger auf diese Kopie.
	/// Die in der Liste verwaltete Klasse muss ebenfalls eine getDevPtr-Methode bereitstellen.
	/// </summary>
	/// <returns>Zeiger auf die Kopie im Device-Speicher</returns>
	CuVector1<T>* getDevPtr();

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

	CU_HSTDEV T& operator[](int pos);

	CU_HSTDEV void set(int pos, T newValue);
	CU_HSTDEV void setAll(T newValue);

	CU_HSTDEV T get(int pos) const;
	CU_HSTDEV T& at(int pos) const;

	/// <summary>
	/// Liefert die Position eines bestimmten Elementes im Vektor oder -1, falls 
	/// das Element nicht im Vektor enthalten ist.
	/// </summary>
	CU_HSTDEV int find(const T &element);

	/// <summary>
	/// Entfernt ein bestimmtes Element aus dem Vektor, indem das Element
	/// mit dem letzten Element des Vektors überschrieben wird.
	/// </summary>
	/// <returns>False, falls das Element nicht im Vektor enthalten ist.</returns>
	CU_HSTDEV bool remove(const T &element);

	/// <summary>
	/// Entfernt ein bestimmtes Element aus dem Vektor, indem das Element
	/// mit dem letzten Element des Vektors überschrieben wird.
	/// </summary>
	/// <returns>False, falls das Element nicht im Vektor enthalten ist.</returns>
	CU_HSTDEV bool remove(int index);

	CU_HSTDEV bool operator==(CuVector1<T> &other);
	CU_HSTDEV bool operator!=(CuVector1<T> &other);

	//CU_HSTDEV void setAll(T value);

	CU_HSTDEV int getSize() { return _size; }

protected:
	T* _data;
	int _size;
	CuVector1<T> *_devicePtr;
};


template<typename T>
CuVector1<T>::CuVector1(int size)
	: _data(0), _size(size), _devicePtr(0)
{
	assert(size > 0);
	_data = new T[size];
	memset(_data, 0, sizeof(T)*size);
}


template<typename T>
CuVector1<T>::CuVector1(const CuVector1<T>& other)
	: CuVector1(other._size)
{
	for (int i = 0; i < _size; i++) {
		_data[i] = other._data[i];
	}
}


template<typename T>
CuVector1<T>::~CuVector1()
{
	if (_data) delete[] _data;
#ifdef __CUDACC__
	if (_devicePtr) {
		CuVector1<T> *tempDevPtr = _devicePtr;
		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuVector1<T>), cudaMemcpyDeviceToHost));
		if (_data) {
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _data));
		}
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, tempDevPtr));
	}
#endif
}


template<typename T>
CuVector1<T>& CuVector1<T>::operator=(const CuVector1<T> &rhs)
{
	if (this != &rhs) {
		if (_data) delete[] _data;
		if (_devicePtr) {
#ifdef __CUDACC__
			CuVector1<T> *tempDevPtr = _devicePtr;
			CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuVector1<T>), cudaMemcpyDeviceToHost));
			if (_data) {
				CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _data));
			}
			CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, tempDevPtr));
#endif
			_devicePtr = 0;
		}
		_size = rhs._size;
		_data = new T[_size];
		for (int i = 0; i < _size; i++) {
			_data[i] = rhs._data[i];
		}
	}
	return *this;
}



template<typename T>
void CuVector1<T>::initialize()
{
	memset(_data, 0, sizeof(T) * _size);
}


template<typename T>
CuVector1<T>* CuVector1<T>::getDevPtr()
{
#ifdef __CUDACC__
	if (!_devicePtr) {
		if (!_data) _data = new T[_size];

		CuVector1<T>* tempObj;
		T* tempData = _data;

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_data, sizeof(T) * _size));

		CUDA_CHECK(cudaMemcpy(_data, tempData, sizeof(T) * _size, cudaMemcpyHostToDevice));

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempObj, sizeof(CuVector1<T>)));

		CUDA_CHECK(cudaMemcpy(tempObj, this, sizeof(CuVector1<T>), cudaMemcpyHostToDevice));

		_devicePtr = tempObj;
		_data = tempData;
	}
#endif
	return _devicePtr;
}


template<typename T>
void CuVector1<T>::copyToHost()
{
#ifdef __CUDACC__
	if (_devicePtr) {
		CuVector1<T>* tempDevObj = _devicePtr;
		T* tempData = _data;
		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuVector1<T>), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(tempData, _data, sizeof(T) * _size, cudaMemcpyDeviceToHost));
		_devicePtr = tempDevObj;
		_data = tempData;
	}
#endif
}


template<typename T>
void CuVector1<T>::copyToDevice()
{
	if (_devicePtr) {
#ifdef __CUDACC__
		CuVector1<T> *tempObj = _devicePtr;
		T* tempData = _data;

		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuVector1<T>), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(_data, tempData, sizeof(T)*_size, cudaMemcpyHostToDevice));

		_data = tempData;
		_devicePtr = tempObj;
#endif
	}
}


template<typename T>
CU_HSTDEV T& CuVector1<T>::operator[](int pos)
{
	assert(pos >= 0);
	assert(pos < _size);
	return _data[pos];
}


template<typename T>
CU_HSTDEV T CuVector1<T>::get(int pos) const
{
	assert(pos >= 0);
	assert(pos < _size);
	return _data[pos];
}


template<typename T>
CU_HSTDEV T& CuVector1<T>::at(int pos) const
{
	assert(pos >= 0);
	assert(pos < _size);
	return _data[pos];
}


template<typename T>
CU_HSTDEV void CuVector1<T>::set(int pos, T newValue)
{
	assert(pos >= 0);
	assert(pos < _size);
	_data[pos] = newValue;
}


template<typename T>
CU_HSTDEV void CuVector1<T>::setAll(T newValue)
{
	for (int i = 0; i < _size; i++) _data[i] = newValue;
}


template<typename T>
CU_HSTDEV bool CuVector1<T>::operator==(CuVector1<T> &other)
{
	for (int i = 0; i < _size; i++) {
		T left = _data[i];
		T right = other._data[i];
		if (left != right) {
			return false;
		}
	}
	return true;
}


template<typename T>
CU_HSTDEV bool CuVector1<T>::operator!=(CuVector1<T> &other)
{
	return !(this->operator==(other));
}


template<typename T>
CU_HSTDEV int CuVector1<T>::find(const T &element)
{
	int i = 0;
	while (i < _size) {
		if (get(i) == element) {
			return i;
		}
		i++;
	}
	return -1;
}


template<typename T>
CU_HSTDEV bool CuVector1<T>::remove(const T &element)
{
	int pos = find(element);
	if (pos >= 0) {
		if (_size > 1) {
			set(pos, get(_size - 1));
		}
		_size--;
		return true;
	}
	return false;
}


template<typename T>
CU_HSTDEV bool CuVector1<T>::remove(int index)
{
	assert(index >= 0);
	assert(index < _size);

	if (index >= 0 && index < _size) {
		if (_size > 1) {
			set(index, get(_size - 1));
		}
		_size--;
		return true;
	}
	return false;
}