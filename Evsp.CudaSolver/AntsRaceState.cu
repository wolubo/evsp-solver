#include "AntsRaceState.h"

#include <assert.h>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/PointInTime.hpp"


CU_HSTDEV AntsRaceState::AntsRaceState(CuProblem *problem, ConsumptionMatrix *consumption, float chargeLimit, bool verbose)
	:_problem(problem), _vehTypeId(), _canCharge(false), _remainingBatteryCapacity(-1.0f),
	_currentStopId(), _currentTime(), _startTime(), _startDepotId(), _consumption(consumption),
	_chargeLimitInPercent(chargeLimit), _chargeLimit(-1.0f),
	_verbose(verbose)
{
}


CU_HSTDEV AntsRaceState::AntsRaceState(const AntsRaceState& other)
	:_problem(other._problem), _vehTypeId(other._vehTypeId), _canCharge(other._canCharge),
	_remainingBatteryCapacity(other._remainingBatteryCapacity), 
	_currentStopId(other._currentStopId), _currentTime(other._currentTime), _startTime(other._startTime),
	_chargeLimitInPercent(other._chargeLimitInPercent), _chargeLimit(other._chargeLimit),
	_startDepotId(other._startDepotId), _consumption(other._consumption), _verbose(other._verbose)
{
}


CU_HSTDEV AntsRaceState::~AntsRaceState()
{
}


CU_HSTDEV void AntsRaceState::startCirculation(StopId startDepotId, VehicleTypeId vehTypeId)
{
	_startDepotId = startDepotId;
	_currentStopId = startDepotId;
	_vehTypeId = vehTypeId;
	//_forceCharge = false;
	_currentTime = PointInTime::invalid();
	_startTime = PointInTime::invalid();
	KilowattHour batteryCapacity = _problem->getVehicleTypes().getVehicleType(_vehTypeId).batteryCapacity;
	_remainingBatteryCapacity = batteryCapacity;
	_chargeLimit = KilowattHour((float)batteryCapacity * _chargeLimitInPercent);
	_canCharge = false;
	if (_verbose) printf("Fahrzeug vorbereitet: stop=%i, vehicle=%i, battery=%.1f\n", (short)_currentStopId, (short)_vehTypeId, (float)_remainingBatteryCapacity);
}


CU_HSTDEV void AntsRaceState::endCirculation()
{
	processEmptyTrip(_startDepotId);

	const CuVehicleType &vt = _problem->getVehicleTypes().getVehicleType(_vehTypeId);

	if (_verbose) printf("Umlauf-Ende: stop=%i, vehicle=%i, battery=%.1f\n", (short)_currentStopId, (short)_vehTypeId, (float)_remainingBatteryCapacity);

	_currentTime = PointInTime(0);
	_currentStopId = StopId::invalid();
	_startDepotId = StopId::invalid();
}


CU_HSTDEV bool AntsRaceState::canProcessEmptyTrip(EmptyTripId emptyTripId)
{
	assert(_problem->getEmptyTrips().getEmptyTrip(emptyTripId).fromStopId == _currentStopId);
	return _consumption->getEmptyTripConsumption(emptyTripId, _vehTypeId) <= _remainingBatteryCapacity;
}


CU_HSTDEV bool AntsRaceState::canProcessEmptyTrip(StopId toStop)
{
	assert(toStop.isValid());
	if (_currentStopId == toStop) return true;
	EmptyTripId etId = _problem->getConnectionMatrix().getEmptyTripId(_currentStopId, toStop);
	return canProcessEmptyTrip(etId);
}


CU_HSTDEV DurationInSeconds AntsRaceState::processEmptyTrip(EmptyTripId emptyTripId)
{
	if (!emptyTripId.isValid())
	{
		return DurationInSeconds::invalid();
	}

	const CuEmptyTrip &emptyTrip = _problem->getEmptyTrips().getEmptyTrip(emptyTripId);

	assert(emptyTrip.fromStopId == _currentStopId);

	DurationInSeconds duration = emptyTrip.duration;
	KilowattHour consumption = _consumption->getEmptyTripConsumption(emptyTripId, _vehTypeId);
	updateRemainingBatteryCapacity(consumption);

	// Die zeitabhängigen Fahrzeugkosten werden am Ende des Umlaufs addiert, wenn die Gesamtdauer des Umlaufs feststeht.

	_currentStopId = emptyTrip.toStopId;

	_currentTime = PointInTime((int)_currentTime + (int)duration);

	if (_verbose) printf("Verbindungsfahrt: %i --> %i, currentTime=%i, vehicle=%i, battery=%.1f\n", (short)emptyTrip.fromStopId, (short)_currentStopId, (int)_currentTime, (short)_vehTypeId, (float)_remainingBatteryCapacity);

	return duration;
}


CU_HSTDEV DurationInSeconds AntsRaceState::processEmptyTrip(StopId toStop)
{
	assert(toStop.isValid());
	EmptyTripId etId = _problem->getConnectionMatrix().getEmptyTripId(_currentStopId, toStop);
	return processEmptyTrip(etId);
}


CU_HSTDEV bool AntsRaceState::canProcessServiceTrip(ServiceTripId servTripId)
{
	/*
	Wenn es sich um die Prüfung für die erste Fahrt eines neuen Umlaufs handelt steht der Fahrzeugtyp noch nicht fest. Da es aber
	keine vorherige Fahrt im Umlauf gibt muss die Servicefahrt möglich sein.
	*/
	if (!_vehTypeId.isValid()) {
		printf("canProcessServiceTrip %i: This is the first trip. So it's possible.\n", (short)servTripId);
		return true;
	}

	const CuServiceTrip &serviceTrip = _problem->getServiceTrips().getServiceTrip(servTripId);

	// Wird eine Verbindungsfahrt benötigt?
	EmptyTripId emptyTripId = _problem->getConnectionMatrix().getEmptyTripId(_currentStopId, serviceTrip.fromStopId);
	assert(emptyTripId.isValid() || _currentStopId == serviceTrip.fromStopId);
	DurationInSeconds emptyTripDuration(0);
	if (emptyTripId.isValid())
	{
		emptyTripDuration = _problem->getEmptyTrips().getEmptyTrip(emptyTripId).duration;
	}

	if (serviceTrip.departure > PointInTime((int)_currentTime + (int)emptyTripDuration)) {
		if (_problem->getVehicleTypeGroups().hasVehicleType(serviceTrip.vehicleTypeGroupId, _vehTypeId)) {

			// Reicht Batterie aus, um die Fahrt durchzuführen? 
			KilowattHour consumption = _consumption->getServiceTripConsumption(servTripId, _vehTypeId);
			if (emptyTripId.isValid()) {
				consumption += _consumption->getEmptyTripConsumption(emptyTripId, _vehTypeId);
			}

			// Reicht Batterie aus, um im Anschluss an die Fahrt notfalls auch das Depot noch erreichen zu können?
			EmptyTripId etToDepotId = _problem->getConnectionMatrix().getEmptyTripId(serviceTrip.toStopId, _startDepotId);
			assert(etToDepotId.isValid() || serviceTrip.toStopId == _startDepotId);
			if (etToDepotId.isValid()) {
				consumption += _consumption->getEmptyTripConsumption(etToDepotId, _vehTypeId);
			}

			if (_remainingBatteryCapacity - consumption > 0.01f) {
				return true;
			}
		}
	}
	return false;
}


CU_HSTDEV void AntsRaceState::processServiceTrip(ServiceTripId servTripId)
{
	const CuServiceTrip &serviceTrip = _problem->getServiceTrips().getServiceTrip(servTripId);

	// Evtl. nötige Verbindungsfahrt berücksichtigen.
	DurationInSeconds et_duration = processEmptyTrip(_problem->getConnectionMatrix().getEmptyTripId(_currentStopId, serviceTrip.fromStopId));
	assert(_currentStopId == serviceTrip.fromStopId);
	assert(_currentTime < serviceTrip.departure);

	updateRemainingBatteryCapacity(_consumption->getServiceTripConsumption(servTripId, _vehTypeId));

	_currentStopId = serviceTrip.toStopId;

	// Die zeitabhängigen Fahrzeugkosten werden am Ende des Umlaufs addiert, wenn die Gesamtdauer des Umlaufs feststeht.
	_currentTime = serviceTrip.arrival;

	// Falls dies die erste SF im Umlauf ist: Startzeitpunkt festhalten.
	if (!_startTime.isValid()) {
		_startTime = PointInTime((int)serviceTrip.departure - (int)et_duration);
	}

	if (_verbose) printf("Servicefahrt: %i --> %i (%i - %i), current time=%i, vehicle=%i, battery=%.1f\n", (short)serviceTrip.fromStopId, (short)_currentStopId, (int)serviceTrip.departure, (int)serviceTrip.arrival, (int)_currentTime, (short)_vehTypeId, (float)_remainingBatteryCapacity);
}


CU_HSTDEV void AntsRaceState::processCharging(EmptyTripId emptyTripId)
{
	processEmptyTrip(emptyTripId); // Falls eine Verbindungsfahrt nötig ist.

	const CuVehicleType &vehType = _problem->getVehicleTypes().getVehicleType(_vehTypeId);

	_currentTime = PointInTime(_currentTime + (int)vehType.rechargingTime);

	// TODO Beim Lösen eines konventionellen VSP dürfen die Ladekosten nicht berechnet werden! --> Kosten durch Config auf 0 setzen!

	_remainingBatteryCapacity = vehType.batteryCapacity;

	_canCharge = false;

	if (_verbose) printf("Aufladen: Station%i, currentTime=%i, vehicle=%i, battery=%.1f\n", (short)_currentStopId, (int)_currentTime, (short)_vehTypeId, (float)_remainingBatteryCapacity);
}


CU_HSTDEV void AntsRaceState::processCharging(StopId chargingStationId)
{
	processCharging(_problem->getConnectionMatrix().getEmptyTripId(_currentStopId, chargingStationId));
}


/*
Der Aufladevorgang ist möglich, wenn die Ladeschwelle unterschritten ist und die restliche Batteriekapazität ausreicht,
die Ladestation zu erreichen.
*/
CU_HSTDEV bool AntsRaceState::canProcessCharging(StopId chargingStationId)
{
	if (!(_canCharge /*|| _forceCharge*/)) return false;
	return canProcessEmptyTrip(chargingStationId);
}


CU_HSTDEV void AntsRaceState::updateRemainingBatteryCapacity(KilowattHour consumption)
{
	assert(consumption >= KilowattHour(0.0f));
	_remainingBatteryCapacity = KilowattHour((float)_remainingBatteryCapacity - (float)consumption);
	assert(_remainingBatteryCapacity >= KilowattHour(0.0f));
	_canCharge = _remainingBatteryCapacity < _chargeLimit;
}

