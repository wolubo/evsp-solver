#pragma once

#include <string>
#include <math.h>
//#include <limits.h>
#include <assert.h>

#include "Typedefs.h"
#include "PointInTime.hpp"
#include "DeviceLaunchParameters.h"

/// <summary>
/// Repräsentiert einen Zeitraum.
/// </summary>
class PeriodOfTime
{
public:
	/// <summary>Erzeugt einen Zeitraum mit maximaler Dauer (0 bis MAXINT Sekunden).</summary>
	CU_HSTDEV PeriodOfTime() : _begin(0), _end(INT_MAX)	{	}

	/// <summary></summary>
	CU_HSTDEV PeriodOfTime(PointInTime begin, PointInTime end);

	/// <summary></summary>
	CU_HSTDEV PeriodOfTime(int begin, int end);

	/// <summary></summary>
	CU_HSTDEV ~PeriodOfTime() {}

	CU_HSTDEV static PeriodOfTime invalid() { return PeriodOfTime(PointInTime::invalid(), PointInTime::invalid()); }
	CU_HSTDEV bool isValid() { return _begin.isValid() && _end.isValid(); }

	bool operator==(const PeriodOfTime &rhs);
	bool operator!=(const PeriodOfTime &rhs);

	/// <summary></summary>
	CU_HSTDEV PointInTime getBegin() const { return _begin; }

	/// <summary></summary>
	CU_HSTDEV PointInTime getEnd() const { return _end; }

	/// <summary>
	/// Dauer in Sekunden.
	/// </summary>
	CU_HSTDEV DurationInSeconds getDuration() const;

	/// <summary>
	/// Liefert die Schnittmenge von 'this' und 't'. Also den Zeitraum, um den sich 'this' und 't' überschneiden.
	/// </summary>
	/// <param name="t"></param>
	/// <returns>Zeitraum, um den sich 'this' und 't' überschneiden. Das Ergebnis kann auch ein "leerer" Zeitraum 
	/// sein (also ein Zeitraum mit der Dauer 0).</returns>
	CU_HSTDEV PeriodOfTime intersection(const PeriodOfTime& t) const;

	/// Prüft, ob sich 'this' und 't' überschneiden.
	/// </summary>
	/// <param name="t"></param>
	/// <returns>true, wenn sich die beiden Zeiträume überschneiden. Sonst false.</returns>
	CU_HSTDEV bool intersecting(const PeriodOfTime& t) const;

	/// <summary>
	/// Prüft, ob 'this' eine Teilmenge von 't' ist (ob also 'this' in 't' enthalten ist).
	/// </summary>
	/// <param name="t"></param>
	/// <returns>True, wenn 'this' in 't' enthalten ist. Sonst false.</returns>
	CU_HSTDEV bool isSubsetOf(const PeriodOfTime& t) const;

	/// <summary>
	/// Prüft, ob 'this' eine echte Teilmenge von 't' ist (ob also 'this' in 't' enthalten ist).
	/// </summary>
	/// <param name="t"></param>
	/// <returns>True, wenn 'this' in 't' enthalten ist. Sonst false.</returns>
	CU_HSTDEV bool isProperSubsetOf(const PeriodOfTime& t) const;

	/// <summary>
	/// Prüft, ob 'this' eine Obermenge von 't' ist (ob also 't' in 'this' enthalten ist).
	/// </summary>
	/// <param name="t"></param>
	/// <returns>True, wenn 't' in 'this' enthalten ist. Sonst false.</returns>
	CU_HSTDEV bool isSupersetOf(const PeriodOfTime& t) const;

	/// <summary>
	/// Prüft, ob 'this' eine echte Obermenge von 't' ist (ob also 't' in 'this' enthalten ist).
	/// </summary>
	/// <param name="t"></param>
	/// <returns>True, wenn 't' in 'this' enthalten ist. Sonst false.</returns>
	CU_HSTDEV bool isProperSupersetOf(const PeriodOfTime& t) const;

	/// <summary>
	/// Liefert den Zeitraum, der zwischen 'this' und 't' liegt.
	/// </summary>
	/// <param name="t"></param>
	/// <returns>Zeitraum zwischen 'this' und 't' oder einen "leeren" Zeitraum (also ein Zeitraum mit der Dauer 0).</returns>
	CU_HSTDEV PeriodOfTime getInterval(const PeriodOfTime& t) const;

	/// <summary>
	/// Liefert den Zeitraum, der durch Vereinigung der Zeiträume 'this' und 't' entsteht (auch dann, wenn sich 'this' und 't' nicht überschneiden).
	/// </summary>
	/// <param name="t"></param>
	/// <returns></returns>
	CU_HSTDEV PeriodOfTime join(const PeriodOfTime t) const;

	std::string toString();

private:
	PointInTime _begin;
	PointInTime _end;
};



inline PeriodOfTime::PeriodOfTime(PointInTime begin, PointInTime end) : _begin(begin), _end(end)
{
	if (_begin > _end) {
		_begin = end;
		_end = begin;
	}
}


inline PeriodOfTime::PeriodOfTime(int begin, int end) : _begin(begin), _end(end)
{
	if (_begin > _end) {
		_begin = PointInTime(end);
		_end = PointInTime(begin);
	}
}


inline bool PeriodOfTime::operator==(const PeriodOfTime &rhs)
{
	if (this == &rhs) return true;

	if (_begin != rhs._begin) return false;
	if (_end != rhs._end) return false;

	return true;

}


inline bool PeriodOfTime::operator!=(const PeriodOfTime &rhs)
{
	return !(*this == rhs);
}


inline DurationInSeconds PeriodOfTime::getDuration() const
{
	assert(_end > _begin);
	return DurationInSeconds((int)_end - (int)_begin);
}

inline PeriodOfTime PeriodOfTime::intersection(const PeriodOfTime& t) const
{
	if (_end < t._begin) return PeriodOfTime::invalid();
	if (t._end < _begin) return PeriodOfTime::invalid();

	PointInTime left, right;
	if (_begin < t._begin) left = t._begin;
	if (t._begin < _begin) left = _begin;
	if (_end > t._end) right = t._end;
	if (t._end > _end) right = _end;

	return PeriodOfTime(left, right);
}

inline bool PeriodOfTime::intersecting(const PeriodOfTime& t) const
{
	if (_end < t._begin) return false;
	if (t._end < _begin) return false;
	return true;
}

inline bool PeriodOfTime::isSubsetOf(const PeriodOfTime& t) const
{
	return (_begin >= t._begin) && (_end <= t._end);
}

inline bool PeriodOfTime::isProperSubsetOf(const PeriodOfTime& t) const
{
	return ((_begin > t._begin) && (_end <= t._end) || (_begin >= t._begin) && (_end < t._end));
}

inline bool PeriodOfTime::isSupersetOf(const PeriodOfTime& t) const
{
	return t.isSubsetOf(*this);
}

inline bool PeriodOfTime::isProperSupersetOf(const PeriodOfTime& t) const
{
	return t.isProperSubsetOf(*this);
}

inline PeriodOfTime PeriodOfTime::getInterval(const PeriodOfTime& t) const
{
	if (_end < t._begin) return PeriodOfTime(_end + 1, t._begin - 1);
	if (t._end < _begin) return PeriodOfTime(t._begin - 1, _end + 1);
	return PeriodOfTime::invalid();
}

inline PeriodOfTime PeriodOfTime::join(const PeriodOfTime t) const
{
	PointInTime left, right;
	if (_begin < t._begin) left = _begin;
	if (t._begin < _begin) left = t._begin;
	if (_end > t._end) right = _end;
	if (t._end > _end) right = t._end;
	return PeriodOfTime(left, right);
}


inline std::string PeriodOfTime::toString()
{
	return "PeriodOfTime: von " + _begin.toString() + " bis " + _end.toString();
}

