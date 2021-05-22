#include <iostream>
#include <fstream>

#include "EmptyTrip.h"
//#include "EVSP.BaseClasses/EvspLimits.h"


using namespace std;
//using namespace boost;


EmptyTrip::EmptyTrip(std::shared_ptr<Stop> fromStop, std::shared_ptr<Stop> toStop,
	DistanceInMeters distance, DurationInSeconds duration,
	std::string fromTime, std::string toTime)
	: _id(0), _distance(distance), _duration(duration), _fromStop(fromStop), _toStop(toStop), _fromTime(fromTime), _toTime(toTime)
{
	if (fromStop == 0) throw  invalid_argument("fromStop");
	if (toStop == 0) throw  invalid_argument("toStop");
}


DistanceInMeters EmptyTrip::getDistance() const
{
	return _distance;
}


DurationInSeconds EmptyTrip::getDuration() const
{
	return _duration;
}


std::shared_ptr<Stop> EmptyTrip::getFromStop() const
{
	return _fromStop;
}


std::shared_ptr<Stop> EmptyTrip::getToStop() const
{
	return _toStop;
}


string EmptyTrip::toString()
{
	return "Verbindungsfahrt von " + _fromStop->getName() + " nach " + _toStop->getName() + " (Dauer: " + to_string((int)_duration) + ", Stecke: " + to_string((int)_distance) + ")";
}


#ifdef _DEBUG
void EmptyTrip::write2file(std::ofstream &txtfile)
{
	//$DEADRUNTIME:FromStopID;ToStopID;FromTime;ToTime;Distance;RunTime;;;;;
	txtfile <<
		_fromStop->getLegacyId() << ";" <<
		_toStop->getLegacyId() << ";" <<
		_fromTime << ";" <<
		_toTime << ";" <<
		(int)_distance << ";" <<
		(int)_duration << ";" <<
		";;;;" << endl;
}
#endif


