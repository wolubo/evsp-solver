#include "CuProblem.h"

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include "CudaCheck.h"

using namespace std;


CuProblem::CuProblem(CuStops *stops, CuEmptyTrips *emptyTrips, CuServiceTrips *serviceTrips, CuVehicleTypes *vehicleTypes, CuVehicleTypeGroups *vehicleTypeGroups, PlattformConfig plattform, bool performChecks)
	: _deviceObject(0)
{
	_serviceTrips = serviceTrips;
	_emptyTrips = emptyTrips;
	_stops = stops;
	_vehicleTypes = vehicleTypes;
	_vehicleTypeGroups = vehicleTypeGroups;
	_connectionMatrix = new ConnectionMatrix(_emptyTrips, _stops->getNumOfStops());
	_prevNextMatrix = new PrevNextMatrix(_serviceTrips, _emptyTrips, _connectionMatrix, plattform, performChecks);
	_chargingMatrix = new ChargingMatrix(*this, plattform, performChecks);
	_serviceTripCost = new ServiceTripCostMatrix(_serviceTrips, _vehicleTypes, _vehicleTypeGroups, plattform);
	_emptyTripCost = new EmptyTripCostMatrix(_emptyTrips, _vehicleTypes, plattform);
}


CuProblem::~CuProblem()
{
	if (_stops) delete _stops;
	if (_emptyTrips) delete _emptyTrips;
	if (_serviceTrips) delete _serviceTrips;
	if (_vehicleTypes) delete _vehicleTypes;
	if (_vehicleTypeGroups) delete _vehicleTypeGroups;
	if (_connectionMatrix) delete _connectionMatrix;
	if (_prevNextMatrix) delete _prevNextMatrix;
	if (_chargingMatrix) delete _chargingMatrix;
	if (_serviceTripCost) delete _serviceTripCost;
	if (_emptyTripCost) delete _emptyTripCost;

	if (_deviceObject) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _deviceObject));
		_deviceObject = 0;
	}
}


CuProblem* CuProblem::getDevPtr()
{
	if (!_deviceObject) {
		CuProblem temp;

		temp._emptyTrips = _emptyTrips->getDevPtr();
		temp._serviceTrips = _serviceTrips->getDevPtr();
		temp._stops = _stops->getDevPtr();
		temp._vehicleTypes = _vehicleTypes->getDevPtr();
		temp._vehicleTypeGroups = _vehicleTypeGroups->getDevPtr();
		temp._connectionMatrix = _connectionMatrix->getDevPtr();
		temp._prevNextMatrix = _prevNextMatrix->getDevPtr();
		temp._chargingMatrix = _chargingMatrix->getDevPtr();
		temp._serviceTripCost = _serviceTripCost->getDevPtr();
		temp._emptyTripCost = _emptyTripCost->getDevPtr();
		temp._deviceObject = 0;

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_deviceObject, sizeof(CuProblem)));
		CUDA_CHECK(cudaMemcpy(_deviceObject, &temp, sizeof(CuProblem), cudaMemcpyHostToDevice));

		temp._emptyTrips = 0;
		temp._serviceTrips = 0;
		temp._stops = 0;
		temp._vehicleTypes = 0;
		temp._vehicleTypeGroups = 0;
		temp._connectionMatrix = 0;
		temp._prevNextMatrix = 0;
		temp._chargingMatrix = 0;
		temp._serviceTripCost = 0;
		temp._emptyTripCost = 0;
	}
	return _deviceObject;
}


void CuProblem::copyToHost()
{
	if (!_deviceObject) {
		assert(false);
	}
}


CU_HSTDEV AmountOfMoney CuProblem::getEmptyTripCosts(StopId departure, StopId destination, VehicleTypeId vehTypeId) const
{
	AmountOfMoney retVal(0);
	EmptyTripId emptyTripId;
	if (departure != destination) {
		emptyTripId = _connectionMatrix->getEmptyTripId(departure, destination);
		if (emptyTripId.isValid()) {
			retVal = _emptyTripCost->getDistanceDependentCosts(emptyTripId, vehTypeId);
		}
		else {
			retVal = AmountOfMoney::invalid();
		}
	}
	return retVal;
}


CU_HSTDEV AmountOfMoney CuProblem::getServiceTripCosts(StopId departure, StopId destination, ServiceTripId servTripId, const CuServiceTrip &servTrip, VehicleTypeId vehTypeId) const
{
	AmountOfMoney retVal = getEmptyTripCosts(departure, servTrip.fromStopId, vehTypeId);
	if (retVal.isValid()) {
		retVal += _serviceTripCost->getDistanceDependentCosts(servTripId, vehTypeId);
	}
	if (retVal.isValid()) {
		retVal += getEmptyTripCosts(servTrip.toStopId, destination, vehTypeId);
	}
	return retVal;
}


CU_HSTDEV AmountOfMoney CuProblem::getServiceTripCosts(StopId departure, StopId destination, ServiceTripId servTripId, VehicleTypeId vehTypeId) const
{
	return getServiceTripCosts(departure, destination, servTripId, _serviceTrips->getServiceTrip(servTripId), vehTypeId);
}


CU_HSTDEV KilowattHour CuProblem::getEmptyTripConsumption(StopId departure, StopId destination, VehicleTypeId vehTypeId) const
{
	KilowattHour retVal(0.0f);
	EmptyTripId emptyTripId;
	if (departure != destination) {
		emptyTripId = _connectionMatrix->getEmptyTripId(departure, destination);
		assert(emptyTripId.isValid());
		retVal = _emptyTripCost->getConsumption(emptyTripId, vehTypeId);
	}
	return retVal;
}


CU_HSTDEV KilowattHour CuProblem::getServiceTripConsumption(StopId departure, StopId destination, ServiceTripId servTripId, const CuServiceTrip &servTrip, VehicleTypeId vehTypeId) const
{
	KilowattHour retVal = getEmptyTripConsumption(departure, servTrip.fromStopId, vehTypeId);
	retVal += _serviceTripCost->getConsumption(servTripId, vehTypeId);
	retVal += getEmptyTripConsumption(servTrip.toStopId, destination, vehTypeId);
	return retVal;
}


CU_HSTDEV KilowattHour CuProblem::getServiceTripConsumption(StopId departure, StopId destination, ServiceTripId servTripId, VehicleTypeId vehTypeId) const
{
	return getServiceTripConsumption(departure, destination, servTripId, _serviceTrips->getServiceTrip(servTripId), vehTypeId);
}
