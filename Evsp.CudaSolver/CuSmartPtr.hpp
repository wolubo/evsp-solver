#pragma once

#include <assert.h>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"

/// <summary>
/// Smart Pointer für Cuda.
/// ACHTUNG: NICHT THREAD-SAFE!
/// </summary>
template<class T>
class CuSmartPtr
{
public:
	CU_HSTDEV CuSmartPtr();
	CU_HSTDEV CuSmartPtr(T* ptr, bool isArray=false);
	CU_HSTDEV CuSmartPtr(const CuSmartPtr &other);
	CU_HSTDEV ~CuSmartPtr();

	CU_HSTDEV CuSmartPtr& operator=(CuSmartPtr& rhs);
	CU_HSTDEV CuSmartPtr& operator=(T* rhs);

	CU_HSTDEV T* get() const;
	CU_HSTDEV void dropOwnership();

	CU_HSTDEV bool isNull() const { return (_ptr == 0); }
	CU_HSTDEV bool isNotNull() const { return (_ptr != 0); }

	CU_HSTDEV static CuSmartPtr NullPtr() { return  CuSmartPtr(); }

	CU_HSTDEV T* operator->() const { return _ptr; }
	CU_HSTDEV T& operator*() { return *_ptr; }

private:
	T *_ptr;
	int *_rc;
	bool _isArray;
};


template<class T>
CuSmartPtr<T>::CuSmartPtr()
	: _ptr(0), _rc(0), _isArray(false)
{
}


template<class T>
CuSmartPtr<T>::CuSmartPtr(T* ptr, bool isArray)
	: _ptr(ptr), _isArray(isArray)
{
	assert(ptr);
	if (ptr) {
		_rc = (int*)malloc(sizeof(int));
		*_rc = 1;
	}
	else {
		_rc = 0;
	}
}


template<class T>
CuSmartPtr<T>::CuSmartPtr(const CuSmartPtr& other)
{
	assert((this != &other));
	if (this == &other) return;

	if (other.isNotNull()) {
		_rc = other._rc;
		(*_rc)++;
		_ptr = other._ptr;
		_isArray = other._isArray;
	}
	else {
		_rc = 0;
		_ptr = 0;
	}
}


template<class T>
CuSmartPtr<T>::~CuSmartPtr()
{
	if (isNotNull()) {
		if (*_rc <= 1) {
			free(_rc);
			if(_isArray)
				delete [] _ptr;
			else
				delete _ptr;
		}
		else {
			(*_rc)--;
		}
	}
}

template<class T>
CuSmartPtr<T>& CuSmartPtr<T>::operator=(CuSmartPtr& rhs)
{
	assert(this != &rhs);
	if (this == &rhs) return *this;

	if (isNotNull()) {
		if (*_rc <= 1) {
			free(_rc);
			delete _ptr;
		}
		else {
			(*_rc)--;
		}
	}

	if (rhs.isNotNull()) {
		_rc = rhs._rc;
		(*_rc)++;
		_ptr = rhs._ptr;
	}
	else {
		_ptr = 0;
		_rc = 0;
	}

	return *this;
}


template<class T>
CuSmartPtr<T>& CuSmartPtr<T>::operator=(T *rhs)
{
	if (isNotNull()) {
		if (*_rc <= 1) {
			free(_rc);
			delete _ptr;
		}
		else {
			(*_rc)--;
		}
	}

	if (rhs) {
		_rc = (int*)malloc(sizeof(int));
		*_rc = 1;
		_ptr = rhs;
	}
	else {
		_ptr = 0;
		_rc = 0;
	}

	return *this;
}


template<class T>
T* CuSmartPtr<T>::get() const
{
	return _ptr;
}

template<class T>
void CuSmartPtr<T>::dropOwnership()
{
	if (isNotNull()) {
		if (*_rc <= 1) {
			free(_rc);
			delete _ptr;
			_rc = 0;
		}
		else {
			(*_rc)--;
		}
		_ptr = 0;
	}
}
