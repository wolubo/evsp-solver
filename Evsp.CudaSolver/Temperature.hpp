#pragma once

#include <assert.h>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"

class Temperature
{
public:
	CU_HSTDEV static Temperature invalid() { return Temperature(); }

	CU_HSTDEV Temperature() : _value(-1.0f), _min(00.0f), _max(00.0f) {}

	CU_HSTDEV Temperature(float start, float min, float max)
		: _value(start), _min(min), _max(max)
	{
		assert(_min > 0.0f);
		assert(_min < _max);
		assert(start >= _min);
		assert(start <= _max);
	}

	CU_HSTDEV Temperature(float start, float min) : Temperature(start, min, start) {}

	CU_HSTDEV Temperature(const Temperature &other) : _value(other._value), _min(other._min), _max(other._max) {}

	CU_HSTDEV ~Temperature() {}

	CU_HSTDEV float maxValue() { return _max; }
	CU_HSTDEV float minValue() { return _min; }

	CU_HSTDEV bool isValid() { return !isInvalid(); }
	CU_HSTDEV bool isInvalid() { return _value < 0.0f; }

	CU_HSTDEV bool isMin() { return _value == _min; }
	CU_HSTDEV bool isMax() { return _value == _max; }

	CU_HSTDEV float getFactor() { return _value / _max; }

	CU_HSTDEV explicit operator int() const { return (int)floor(_value+0.5f); }
	CU_HSTDEV explicit operator float() const { return _value; }
	CU_HSTDEV explicit operator double() const { return (double)_value; }

	Temperature& operator=(const Temperature& rhs) {
		if (*this == rhs) return *this;
		this->_value = rhs._value;
		return *this;
	}

	Temperature& operator=(float rhs) {
		if (rhs < _min) {
			rhs = _min;
		}
		else if (rhs > _max) {
			rhs = _max;
		}
		this->_value = rhs;
		return *this;
	}

	Temperature& operator+=(const Temperature& rhs) {
		//if (*this == rhs) return *this;
		float newValue = this->_value + rhs._value;
		if (newValue < _min) {
			newValue = _min;
		}
		else if (newValue > _max) {
			newValue = _max;
		}
		this->_value = newValue;
		return *this;
	}

	Temperature& operator-=(const Temperature& rhs) {
		//if (*this == rhs) return *this;
		float newValue = this->_value - rhs._value;
		if (newValue < _min) {
			newValue = _min;
		}
		else if (newValue > _max) {
			newValue = _max;
		}
		this->_value = newValue;
		return *this;
	}

	Temperature& operator+=(float rhs) {
		float newValue = this->_value + rhs;
		if (newValue < _min) {
			newValue = _min;
		}
		else if (newValue > _max) {
			newValue = _max;
		}
		this->_value = newValue;
		return *this;
	}

	Temperature& operator-=(float rhs) {
		float newValue = this->_value - rhs;
		if (newValue < _min) {
			newValue = _min;
		}
		else if (newValue > _max) {
			newValue = _max;
		}
		this->_value = newValue;
		return *this;
	}

	Temperature& operator*=(float rhs) {
		float newValue = (float)this->_value * rhs;
		if (newValue < _min) {
			newValue = _min;
		}
		else if (newValue > _max) {
			newValue = _max;
		}
		this->_value = newValue;
		return *this;
	}

	Temperature& operator/=(float rhs) {
		float newValue = (float)this->_value / (float)rhs;
		if (newValue < _min) {
			newValue = _min;
		}
		else if (newValue > _max) {
			newValue = _max;
		}
		this->_value = newValue;
		return *this;
	}

	//CU_HSTDEV Temperature& operator++() { if (_value < _max) ++_value; return *this; } // Prefix
	//CU_HSTDEV Temperature operator++(int) { Temperature r(*this); if (_value < _max) _value++; return r; } // Postfix
	//CU_HSTDEV Temperature& operator--() { if (_value > _min) --_value; return *this; } // Prefix
	//CU_HSTDEV Temperature operator--(int) { Temperature r(*this); if (_value > _min) _value--; return r; } // Postfix

	CU_HSTDEV friend bool operator==(const Temperature &a, const Temperature &b) { return a._value == b._value; }
	CU_HSTDEV friend bool operator==(const Temperature &a, const float &b) { return a._value == b; }
	CU_HSTDEV friend bool operator!=(const Temperature &a, const Temperature &b) { return a._value != b._value; }
	CU_HSTDEV friend bool operator!=(const Temperature &a, const float &b) { return a._value != b; }
	CU_HSTDEV friend bool operator<=(const Temperature &a, const Temperature &b) { return a._value <= b._value; }
	CU_HSTDEV friend bool operator<=(const Temperature &a, const float &b) { return a._value <= b; }
	CU_HSTDEV friend bool operator>=(const Temperature &a, const Temperature &b) { return a._value >= b._value; }
	CU_HSTDEV friend bool operator>=(const Temperature &a, const float &b) { return a._value >= b; }
	CU_HSTDEV friend bool operator<(const Temperature &a, const Temperature &b) { return a._value < b._value; }
	CU_HSTDEV friend bool operator<(const Temperature &a, const float &b) { return a._value < b; }
	CU_HSTDEV friend bool operator>(const Temperature &a, const Temperature &b) { return a._value > b._value; }
	CU_HSTDEV friend bool operator>(const Temperature &a, const float &b) { return a._value > b; }

private:
	float _value;
	float _max;
	float _min;
};

