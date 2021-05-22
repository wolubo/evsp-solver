#include "CuVehicleTypes.h"

#include <assert.h>
#include <stdexcept> 
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h> 

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"

#include "CudaCheck.h"



CuVehicleTypes::CuVehicleTypes(int maxNumOfVehicleTypes)
	: _numOfVehicleTypes(0), _devicePtr(0)
{
	if (maxNumOfVehicleTypes > Max_VehicleTypes) throw new std::invalid_argument("maxNumOfVehicleTypes > Max_VehicleTypes");
	_vehicleTypes = new CuVector1<CuVehicleType>(maxNumOfVehicleTypes);
}


CuVehicleTypes::~CuVehicleTypes()
{
	if (_vehicleTypes) delete _vehicleTypes;
	if (_devicePtr)
	{
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


CuVehicleTypes* CuVehicleTypes::getDevPtr()
{
	if (!_devicePtr)
	{
		CuVehicleTypes temp;
		temp._devicePtr = 0;
		temp._numOfVehicleTypes = _numOfVehicleTypes;
		temp._vehicleTypes = _vehicleTypes->getDevPtr();
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuVehicleTypes)));
		
		CUDA_CHECK(cudaMemcpy(_devicePtr, &temp, sizeof(CuVehicleTypes), cudaMemcpyHostToDevice));
		temp._vehicleTypes = 0;
	}
	return _devicePtr;
}


void CuVehicleTypes::add(AmountOfMoney vehCost, AmountOfMoney kmCost, AmountOfMoney hourCost, KilowattHour batteryCapacity, KilowattHour consumptionPerServiceJourneyKm, KilowattHour consumptionPerDeadheadKm, AmountOfMoney rechargingCost, DurationInSeconds rechargingTime)
{
	if (_numOfVehicleTypes >= Max_VehicleTypes) throw std::runtime_error("Die maximale Anzahl von Fahrzeugtypen ist überschritten!");

	assert(vehCost < AmountOfMoney(10000000));	// Ein Fahrzeug, dass 10.000.000 Mio. kostet ist nicht plausibel.
	assert(kmCost < AmountOfMoney(1000));		// Kosten von 1.000 pro Kilometer sind nicht plausibel.
	assert(hourCost < AmountOfMoney(1000));
	assert(batteryCapacity > KilowattHour(0.0f));
	//assert(batteryCapacity<1000000.0); Möglich, wenn konventionelles VSP gelöst wird.
	//assert(consumptionPerServiceJourneyKm>KilowattHour(0.0f));
	assert(consumptionPerServiceJourneyKm < KilowattHour(100000.0f));
	//assert(consumptionPerDeadheadKm>KilowattHour(0.0f));
	assert(consumptionPerDeadheadKm < KilowattHour(100000.0f));
	assert(rechargingCost < AmountOfMoney(30000));
	assert(rechargingTime < DurationInSeconds(86400));

	CuVehicleType &vt = (*_vehicleTypes)[_numOfVehicleTypes];
	vt.vehCost = vehCost;
	vt.kmCost = kmCost;
	vt.hourCost = hourCost;
	vt.batteryCapacity = batteryCapacity;
	vt.consumptionPerServiceJourneyKm = consumptionPerServiceJourneyKm;
	vt.consumptionPerDeadheadKm = consumptionPerDeadheadKm;
	vt.rechargingCost = rechargingCost;
	vt.rechargingTime = rechargingTime;

	_numOfVehicleTypes++;
}


CU_DEV int CuVehicleTypes::getNumOfVehicleTypes() const
{
	return _numOfVehicleTypes;
}


CU_HSTDEV CuVehicleType& CuVehicleTypes::getVehicleType(VehicleTypeId vehicleTypeId) const
{
	assert(vehicleTypeId.isValid());
	short s_id = (short)vehicleTypeId;
	assert(s_id < _numOfVehicleTypes);
	return (*_vehicleTypes)[s_id];
}


