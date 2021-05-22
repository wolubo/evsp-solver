#pragma once

#include <assert.h>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"

/// <summary>
/// Verwaltet Bit-Werte in einem Vektor fixer Größe. 
/// Optimiert für einen möglichst geringen Speicherbedarf.
/// Nicht für konkurierende Zugriffe geeignet (nicht thread-safe)!
/// </summary>
template<typename TYPE, int SIZE>
class CuBoolVector2
{
public:
	CU_HSTDEV CuBoolVector2();
	CU_HSTDEV CuBoolVector2(const CuBoolVector2 &other) { assert(false); }
	CU_HSTDEV ~CuBoolVector2() {}
	CuBoolVector2& operator=(const CuBoolVector2 &rhs) { assert(false); return *this; }
	CU_HSTDEV void setAll();
	CU_HSTDEV void unsetAll();
	CU_HSTDEV void set(int index);
	CU_HSTDEV void unset(int index);
	CU_HSTDEV void toggle(int index);
	CU_HSTDEV bool get(int index);
protected:
	static const int _arraySize = (SIZE + sizeof(TYPE) - 1) / sizeof(TYPE);
	TYPE _data[_arraySize];
};


template<typename TYPE, int SIZE>
CU_HSTDEV CuBoolVector2<TYPE, SIZE>::CuBoolVector2()
{
	memset(&_data, 0, sizeof(_data));
}


template<typename TYPE, int SIZE>
CU_HSTDEV void CuBoolVector2<TYPE, SIZE>::setAll()
{
	for (int i = 0; i < _arraySize; i++) _data[i] = ~(0u);
}


template<typename TYPE, int SIZE>
CU_HSTDEV void CuBoolVector2<TYPE, SIZE>::unsetAll()
{
	for (int i = 0; i < _arraySize; i++) _data[i] = 0u;
}


template<typename TYPE, int SIZE>
CU_HSTDEV void CuBoolVector2<TYPE, SIZE>::set(int index)
{
	int arrayIndex = index / sizeof(TYPE);
	int bitIndex = index % sizeof(TYPE);
	assert(arrayIndex < _arraySize);
	_data[arrayIndex] |= 1u << bitIndex; // or
}


template<typename TYPE, int SIZE>
CU_HSTDEV void CuBoolVector2<TYPE, SIZE>::unset(int index)
{
	int arrayIndex = index / sizeof(TYPE);
	int bitIndex = index % sizeof(TYPE);
	assert(arrayIndex < _arraySize);
	_data[arrayIndex] &= ~(1u << bitIndex); // and
}


template<typename TYPE, int SIZE>
CU_HSTDEV void CuBoolVector2<TYPE, SIZE>::toggle(int index)
{
	int arrayIndex = index / sizeof(TYPE);
	int bitIndex = index % sizeof(TYPE);
	assert(arrayIndex < _arraySize);
	_data[arrayIndex] ^= 1u << bitIndex; // xor
}


template<typename TYPE, int SIZE>
CU_HSTDEV bool CuBoolVector2<TYPE, SIZE>::get(int index)
{
	int arrayIndex = index / sizeof(TYPE);
	int bitIndex = index % sizeof(TYPE);
	assert(arrayIndex < _arraySize);
	int value = _data[arrayIndex] & (1u << bitIndex);
	return value != 0;
}


