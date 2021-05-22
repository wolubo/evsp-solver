#pragma once

#include <iostream>  

template<class ID_CLASS>
class SafeFloatType
{
public:

	CU_HSTDEV SafeFloatType() { _value = 0.0f; }

	CU_HSTDEV explicit SafeFloatType(float val) : _value(val) { }

	CU_HSTDEV explicit operator float() const { return _value; }

	CU_HSTDEV SafeFloatType& operator=(const SafeFloatType &rhs)
	{
		if (this != &rhs) {
			_value = rhs._value;
		}
		return *this;
	}

	CU_HSTDEV const SafeFloatType operator+(const SafeFloatType &rhs) const { return SafeFloatType(_value + rhs._value); }
	CU_HSTDEV const SafeFloatType operator+(const float &rhs) const { return SafeFloatType(_value + rhs); }
	CU_HSTDEV const SafeFloatType operator-(const SafeFloatType &rhs) const { return SafeFloatType(_value - rhs._value); }
	CU_HSTDEV const SafeFloatType operator-(const float &rhs) const { return SafeFloatType(_value - rhs); }

	CU_HSTDEV SafeFloatType& operator+=(const SafeFloatType& rhs) {
		if (this != &rhs) {
			this->_value += rhs._value;
		}
		return *this;
	}

	CU_HSTDEV SafeFloatType& operator-=(const SafeFloatType& rhs) {
		if (this != &rhs) {
			this->_value -= rhs._value;
		}
		return *this;
	}

	CU_HSTDEV SafeFloatType& operator+=(float rhs) {
		this->_value += rhs;
		return *this;
	}

	CU_HSTDEV SafeFloatType& operator-=(float rhs) {
		this->_value -= rhs;
		return *this;
	}

	CU_HSTDEV SafeFloatType& operator*=(float rhs) {
		this->_value *= (int)rhs;
		return *this;
	}

	CU_HSTDEV SafeFloatType& operator/=(float rhs) {
		this->_value /= (int)rhs;
		return *this;
	}

	CU_HSTDEV friend bool operator==(const SafeFloatType &a, const SafeFloatType &b) { return a._value == b._value; }
	CU_HSTDEV friend bool operator!=(const SafeFloatType &a, const SafeFloatType &b) { return a._value != b._value; }
	CU_HSTDEV friend bool operator<=(const SafeFloatType &a, const SafeFloatType &b) { return a._value <= b._value; }
	CU_HSTDEV friend bool operator<=(const SafeFloatType &a, const float &b) { return a._value <= b; }
	CU_HSTDEV friend bool operator>=(const SafeFloatType &a, const SafeFloatType &b) { return a._value >= b._value; }
	CU_HSTDEV friend bool operator>=(const SafeFloatType &a, const float &b) { return a._value >= b; }
	CU_HSTDEV friend bool operator<(const SafeFloatType &a, const SafeFloatType &b) { return a._value < b._value; }
	CU_HSTDEV friend bool operator<(const SafeFloatType &a, const float &b) { return a._value < b; }
	CU_HSTDEV friend bool operator>(const SafeFloatType &a, const SafeFloatType &b) { return a._value > b._value; }
	CU_HSTDEV friend bool operator>(const SafeFloatType &a, const float &b) { return a._value > b; }

private:
	float _value;
};

