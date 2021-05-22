#pragma once

#include <assert.h>
#include <string>
#include <vector>
#include <stdexcept>
#include "Typedefs.h"
#include "DeviceLaunchParameters.h"


class PointInTime
{
public:
	CU_HSTDEV PointInTime() : _numberOfSeconds(INT_MAX) {}
	CU_HSTDEV explicit PointInTime(int numberOfSeconds) : _numberOfSeconds(numberOfSeconds) {}
	CU_HSTDEV PointInTime(int days, int hours, int minutes, int seconds) { _numberOfSeconds = days * 86400 + hours * 3600 + minutes * 60 + seconds; }
	CU_HSTDEV ~PointInTime() {}

	CU_HSTDEV static PointInTime invalid() { return PointInTime(INT_MAX); }
	CU_HSTDEV bool isValid() { return _numberOfSeconds != INT_MAX; }

	/// <summary>
	/// Tag im Intervall [0..999]
	/// </summary>
	CU_HSTDEV int getDay() const { return _numberOfSeconds / 86400; }// Jeder Tag hat 86400 Sekunden.

	/// <summary>
	/// Stunde im Intervall [0..23]
	/// </summary>
	CU_HSTDEV int getHour() const { return (_numberOfSeconds / 3600) - (getDay() * 24); }

	/// <summary>
	/// Minute im Intervall [0..59]
	/// </summary>
	CU_HSTDEV int getMinute() const { return (_numberOfSeconds / 60) - (getHour() * 60) - (getDay() * 1440); }

	/// <summary>
	/// Sekunde im Intervall [0..59]
	/// </summary>
	CU_HSTDEV int getSecond() const { return _numberOfSeconds % 60; }

	CU_HSTDEV friend bool operator==(const PointInTime &a, const PointInTime &b) { return a._numberOfSeconds == b._numberOfSeconds; }
	CU_HSTDEV friend bool operator!=(const PointInTime &a, const PointInTime &b) { return a._numberOfSeconds != b._numberOfSeconds; }
	CU_HSTDEV friend bool operator<=(const PointInTime &a, const PointInTime &b) { return a._numberOfSeconds <= b._numberOfSeconds; }
	CU_HSTDEV friend bool operator>=(const PointInTime &a, const PointInTime &b) { return a._numberOfSeconds >= b._numberOfSeconds; }
	CU_HSTDEV friend bool operator<(const PointInTime &a, const PointInTime &b) { return a._numberOfSeconds < b._numberOfSeconds; }
	CU_HSTDEV friend bool operator>(const PointInTime &a, const PointInTime &b) { return a._numberOfSeconds > b._numberOfSeconds; }

	CU_HSTDEV PointInTime operator+(const PointInTime& right) const{		return PointInTime(_numberOfSeconds + right._numberOfSeconds);	}
	CU_HSTDEV PointInTime operator+(int right) const{		return PointInTime(_numberOfSeconds + right);	}
	CU_HSTDEV PointInTime operator-(const PointInTime right) const{		return PointInTime(_numberOfSeconds - right._numberOfSeconds);	}
	CU_HSTDEV PointInTime operator-(int right) const{		return PointInTime(_numberOfSeconds - right);	}
	CU_HSTDEV PointInTime& operator=(const PointInTime& right);

	CU_HSTDEV PointInTime& operator++() { ++_numberOfSeconds; return *this; } // Prefix
	CU_HSTDEV PointInTime operator++(int) { PointInTime r(_numberOfSeconds); _numberOfSeconds++; return r; } // Postfix

	/// <summary>
	/// Anzahl der Sekunden seit 000:00:00:00.
	/// </summary>
	CU_HSTDEV explicit operator int() const { return _numberOfSeconds; }

	std::string toString();

private:
	int _numberOfSeconds;
};


inline PointInTime& PointInTime::operator=(const PointInTime& right)
{
	if (this != &right) {
		_numberOfSeconds = right._numberOfSeconds;
	}
	return *this;
}


inline void formatter(std::string &str, int number, int digits)
{
	if (digits == 3)
	{
		if (number >= 100) str = std::to_string(number);
		else if (number >= 10) str = "0" + std::to_string(number);
		else str = "00" + std::to_string(number);
	}
	else if (digits == 2)
	{
		assert(number < 100);
		if (number >= 10) str = std::to_string(number);
		else str = "0" + std::to_string(number);
	}
	else
	{
		assert(false);
	}
}

inline std::string PointInTime::toString()
{
	std::string retVal = "", s;

	if (getDay() > 0)
	{
		formatter(retVal, getDay(), 3);
		retVal += ":";
	}
	else {
		//retVal += "000:";
	}

	formatter(s, getHour(), 2);
	retVal += s + ":";

	formatter(s, getMinute(), 2);
	retVal += s + ":";

	formatter(s, getSecond(), 2);
	retVal += s;

	return retVal;
}

