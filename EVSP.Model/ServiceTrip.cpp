#include <iostream>
#include <fstream>

#include "ServiceTrip.h"
#include "BusRoute.h"
//#include "EVSP.BaseClasses/EvspLimits.h"

using namespace std;
//using namespace boost;
//using namespace WoBo::EVSP::BaseClasses;



ServiceTrip::ServiceTrip(
	const std::string& legacyId,
	std::shared_ptr<BusRoute> route,
	std::shared_ptr<Stop>  fromStop,
	std::shared_ptr<Stop>  toStop,
	PointInTime depTime, PointInTime arrTime,
	std::shared_ptr<VehicleTypeGroup> vehTypeGroup,
	DistanceInMeters distance,
	std::string minAheadTime, std::string minLayoverTime, std::string maxShiftBackwardSeconds, std::string maxShiftForwardSeconds)
	: _legacyId(legacyId), _route(route), _fromStop(fromStop), _toStop(toStop), _scheduledTime(depTime, arrTime), _vehicleTypeGroup(vehTypeGroup), _distance(distance), _minAheadTime(minAheadTime), _minLayoverTime(minLayoverTime), _maxShiftBackwardSeconds(maxShiftBackwardSeconds), _maxShiftForwardSeconds(maxShiftForwardSeconds)
{
	if (route == 0) throw  invalid_argument("route");
	if (fromStop == 0) throw  invalid_argument("fromStop");
	if (toStop == 0) throw  invalid_argument("toStop");
	if (depTime >= arrTime) throw  invalid_argument("Ankunftszeit an der Zielhaltestelle liegt vor der Abfahrtszeit an der Starthaltestelle!");
	if (vehTypeGroup == 0) throw  invalid_argument("vehTypeGroup");
	if (distance == 0) throw  invalid_argument("Die Distanz zwischen zwei Haltestellen kann nicht 0 sein.");
}

PeriodOfTime ServiceTrip::getScheduledTime() const
{
	return _scheduledTime;
}


DistanceInMeters ServiceTrip::getDistance() const
{
	return _distance;
}


std::shared_ptr<Stop> ServiceTrip::getFromStop() const
{
	return _fromStop;
}


std::shared_ptr<Stop> ServiceTrip::getToStop() const
{
	return _toStop;
}


std::shared_ptr<BusRoute> ServiceTrip::getRoute() const
{
	return _route;
}


std::shared_ptr<VehicleTypeGroup> ServiceTrip::getVehicleTypeGroup() const
{
	return _vehicleTypeGroup;
}


bool ServiceTrip::isRoundtrip()
{
	return (_fromStop == _toStop);
}


std::string ServiceTrip::toString()
{
	return "Servicefahrt der Linie " + _route->getName() + " von " + _fromStop->getName() + " (" + _scheduledTime.getBegin().toString() + ") nach " + _toStop->getName() + " (" + _scheduledTime.getEnd().toString() + ")";
}


void ServiceTrip::write2file(std::ofstream &txtfile)
{
	//$SERVICEJOURNEY:ID;LineID;FromStopID;ToStopID;DepTime;ArrTime;MinAheadTime;MinLayoverTime;VehTypeGroupID;MaxShiftBackwardSeconds; MaxShiftForwardSeconds;Distance
	txtfile << _legacyId << ";" <<
		_route->getLegacyId() << ";" <<
		_fromStop->getLegacyId() << ";" <<
		_toStop->getLegacyId() << ";" <<
		_scheduledTime.getBegin().toString() << ";" <<
		_scheduledTime.getEnd().toString() << ";" <<
		_minAheadTime << ";" <<
		_minLayoverTime << ";" <<
		_vehicleTypeGroup->getLegacyId() << ";" <<
		_maxShiftBackwardSeconds << ";" <<
		_maxShiftForwardSeconds << ";" <<
		(int)_distance << endl;
}

