#pragma once

#include "DeviceLaunchParameters.h"

/// <summary>
/// Implementiert Datentypen, die der Compiler prüfen kann. Auf diese Weise können Bugs, die durch das Verwechseln von Attributen, die auf 
/// C++Basistypen beruhen ausgeschlossen werden.
/// ID_CLASS:	Frei wählbarer Bezeichner, der dafür sorgt, dass der Compiler die verschiedenen Datentypen unterscheiden kann. 
///				Wird im Template nicht verwendet.
/// TYPE:		C++-Basistyp, auf dem der Datentyp beruht.
/// INVALID_VALUE:	Ein Wert des Datentyps, der als Marker für ungültige Werte verwendet wird.
/// </summary>
template<class ID_CLASS, typename TYPE, TYPE INVALID_VALUE>
class SafeType
{
public:
	CU_HSTDEV static SafeType invalid() { return SafeType(); }

	CU_HSTDEV SafeType() { _value = INVALID_VALUE; }

	CU_HSTDEV explicit SafeType(TYPE val) : _value(val) { }

	CU_HSTDEV bool isValid() const { return _value != INVALID_VALUE; }

	CU_HSTDEV explicit operator TYPE() const { return _value; }
	CU_HSTDEV explicit operator float() const { return (float)_value; }
	CU_HSTDEV explicit operator double() const { return (double)_value; }

	CU_HSTDEV SafeType& operator=(const SafeType &rhs)
	{
		if (this != &rhs) {
			_value = rhs._value;
		}
		return *this;
	}

	CU_HSTDEV SafeType& operator++() { ++_value; return *this; } // Prefix
	CU_HSTDEV SafeType operator++(int) { SafeType r(_value); _value++; return r; } // Postfix
	CU_HSTDEV SafeType& operator--() { --_value; return *this; } // Prefix
	CU_HSTDEV SafeType operator--(int) { SafeType r(_value); _value--; return r; } // Postfix

	CU_HSTDEV const SafeType operator+(const SafeType &rhs) const { return SafeType(_value + rhs._value); }
	CU_HSTDEV const SafeType operator+(const TYPE &rhs) const { return SafeType(_value + rhs); }
	CU_HSTDEV const SafeType operator-(const SafeType &rhs) const { return SafeType(_value - rhs._value); }
	CU_HSTDEV const SafeType operator-(const TYPE &rhs) const { return SafeType(_value - rhs); }

	CU_HSTDEV SafeType& operator+=(const SafeType& rhs) {
		if (this != &rhs) {
			this->_value += rhs._value;
		}
		return *this;
	}

	CU_HSTDEV SafeType& operator-=(const SafeType& rhs) {
		if (this != &rhs) {
			this->_value -= rhs._value;
		}
		return *this;
	}

	CU_HSTDEV SafeType& operator+=(int rhs) {
		this->_value += rhs;
		return *this;
	}

	CU_HSTDEV SafeType& operator-=(int rhs) {
		this->_value -= rhs;
		return *this;
	}

	CU_HSTDEV SafeType& operator*=(float rhs) {
		this->_value *= (int)rhs;
		return *this;
	}

	CU_HSTDEV SafeType& operator/=(float rhs) {
		this->_value /= (int)rhs;
		return *this;
	}

	CU_HSTDEV friend bool operator==(const SafeType &a, const SafeType &b) { return a._value == b._value; }
	CU_HSTDEV friend bool operator==(const SafeType &a, const TYPE &b) { return a._value == b; }
	CU_HSTDEV friend bool operator!=(const SafeType &a, const SafeType &b) { return a._value != b._value; }
	CU_HSTDEV friend bool operator!=(const SafeType &a, const TYPE &b) { return a._value != b; }
	CU_HSTDEV friend bool operator<=(const SafeType &a, const SafeType &b) { return a._value <= b._value; }
	CU_HSTDEV friend bool operator<=(const SafeType &a, const TYPE &b) { return a._value <= b; }
	CU_HSTDEV friend bool operator>=(const SafeType &a, const SafeType &b) { return a._value >= b._value; }
	CU_HSTDEV friend bool operator>=(const SafeType &a, const TYPE &b) { return a._value >= b; }
	CU_HSTDEV friend bool operator<(const SafeType &a, const SafeType &b) { return a._value < b._value; }
	CU_HSTDEV friend bool operator<(const SafeType &a, const TYPE &b) { return a._value < b; }
	CU_HSTDEV friend bool operator>(const SafeType &a, const SafeType &b) { return a._value > b._value; }
	CU_HSTDEV friend bool operator>(const SafeType &a, const TYPE &b) { return a._value > b; }

private:
	TYPE _value;
};

