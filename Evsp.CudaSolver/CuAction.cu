#include "CuAction.h"
#include <assert.h>


 CuAction::CuAction()
	: _type(CuActionType::INVALID_ACTION), _serviceTripId(), _fromStopId(), _toStopId(),
	_period(), _distance()
{
}


 CuAction::CuAction(const CuAction &other)
	: _type(other._type), _serviceTripId(other._serviceTripId), _fromStopId(other._fromStopId), _toStopId(other._toStopId),
	_period(other._period), _distance(other._distance)
{
}


// Erzeugt eine gültige Aktion des Typs VISIT_DEPOT.
 CuAction::CuAction(StopId stopId)
	: _type(CuActionType::VISIT_DEPOT), _serviceTripId(), _fromStopId(stopId), _toStopId(stopId), _period(0, 0), _distance(0)
{
	assert(stopId.isValid());
	assert(!_serviceTripId.isValid());
}


// Erzeugt eine gültige Aktion des Typs CHARGING.
 CuAction::CuAction(StopId stopId, const CuVehicleType &vehType)
	: _type(CuActionType::CHARGE), _serviceTripId(), _fromStopId(stopId),
	_toStopId(stopId), _period(0, (int)vehType.rechargingTime), _distance(0)
{
	assert(stopId.isValid());
	assert(!_serviceTripId.isValid());
	assert(_period.getDuration() == vehType.rechargingTime);
}


// Erzeugt eine gültige Aktion des Typs EMPTY_TRIP.
 CuAction::CuAction(const CuEmptyTrip &emptyTrip)
	: _type(CuActionType::EMPTY_TRIP), _serviceTripId(), _fromStopId(emptyTrip.fromStopId), _toStopId(emptyTrip.toStopId), _period(0, (int)emptyTrip.duration), _distance(emptyTrip.distance)
{
	assert(!_serviceTripId.isValid());
}


// Erzeugt eine gültige Aktion des Typs CuActionType::SERVICE_TRIP.
 CuAction::CuAction(ServiceTripId serviceTripId, const CuServiceTrip &serviceTrip)
	: _type(CuActionType::SERVICE_TRIP), _serviceTripId(serviceTripId), _fromStopId(serviceTrip.fromStopId),
	_toStopId(serviceTrip.toStopId), _period(serviceTrip.departure, serviceTrip.arrival), _distance(serviceTrip.distance)
{
	assert(_serviceTripId.isValid());
}


CuAction& CuAction::operator=(const CuAction &rhs)
{
	if (this != &rhs) {
		_type = rhs._type;
		_serviceTripId = rhs._serviceTripId;
		_fromStopId = rhs._fromStopId;
		_toStopId = rhs._toStopId;
		_period = rhs._period;
		_distance = rhs._distance;
	}
	return *this;
}


bool CuAction::operator==(const CuAction &rhs)
{
	if (this == &rhs) return true;

	if (_type != rhs._type) return false;
	if (_serviceTripId != rhs._serviceTripId) return false;
	if (_fromStopId != rhs._fromStopId) return false;
	if (_toStopId != rhs._toStopId) return false;
	if (_period != rhs._period) return false;
	if (_distance != rhs._distance) return false;

	return true;
}


bool CuAction::operator!=(const CuAction &rhs)
{
	return !(*this == rhs);
}

ServiceTripId CuAction::getServiceTripId() const
{
	return _serviceTripId;
}


StopId CuAction::getChargingStationId() const
{
	if (_type == CuActionType::CHARGE) {
		return _fromStopId;
	}
	return StopId::invalid();
}


 StopId CuAction::getDepotId() const
{
	if (_type == CuActionType::VISIT_DEPOT) {
		return _fromStopId;
	}
	return StopId::invalid();
}


 StopId CuAction::getStopId() const
{
	if (_type == CuActionType::VISIT_DEPOT || _type == CuActionType::CHARGE) {
		return _fromStopId;
	}
	return StopId::invalid();
}


 void CuAction::dump() const
{
	switch (_type) {
	case CuActionType::CHARGE:
		printf("C(%i)", (short)_fromStopId);
		break;
	case CuActionType::EMPTY_TRIP:
		printf("ET(%i to %i)", (short)_fromStopId, (short)_toStopId);
		break;
	case CuActionType::INVALID_ACTION:
		printf("<invalid>");
		break;
	case CuActionType::SERVICE_TRIP:
		printf("ST %i (%i to %i)", (short)_serviceTripId, (short)_fromStopId, (short)_toStopId);
		break;
	case CuActionType::VISIT_DEPOT:
		printf("D(%i)", (short)_fromStopId);
		break;
	default:
		printf("<unknown>");
	}
}


 void CuAction::dumpTime() const
{
	switch (_type) {
	case CuActionType::CHARGE:
		printf("C(%i, %i)", (short)_fromStopId, (int)_period.getDuration());
		break;
	case CuActionType::EMPTY_TRIP:
		printf("ET(%i to %i, %i)", (short)_fromStopId, (short)_toStopId, (int)_period.getDuration());
		break;
	case CuActionType::INVALID_ACTION:
		printf("<invalid>");
		break;
	case CuActionType::SERVICE_TRIP:
		printf("ST(%i to %i, %i-%i)", (short)_fromStopId, (short)_toStopId, (int)_period.getBegin(), (int)_period.getEnd());
		break;
	case CuActionType::VISIT_DEPOT:
		printf("D(%i, %i)", (short)_fromStopId, (int)_period.getDuration());
		break;
	default:
		printf("<unknown>");
	}
}
