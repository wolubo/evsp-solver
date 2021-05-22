#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuServiceTrips.h"
#include "CuEmptyTrips.h"
#include "CuVehicleTypes.h"
#include "CuMatrix1.hpp"


class ConsumptionMatrix
{
public:
	ConsumptionMatrix(CuEmptyTrips &emptyTrips, CuServiceTrips &serviceTrips, CuVehicleTypes &vehicleTypes, PlattformConfig plattform);
	~ConsumptionMatrix();

	ConsumptionMatrix* getDevPtr();

	CU_HSTDEV KilowattHour getEmptyTripConsumption(EmptyTripId emptyTripId, VehicleTypeId vehTypeId);
	CU_HSTDEV KilowattHour getServiceTripConsumption(ServiceTripId servTripId, VehicleTypeId vehTypeId);

private:
	void createOnCpu(CuEmptyTrips &emptyTrips, int numOfEmptyTrips, CuServiceTrips &serviceTrips, int numOfServiceTrips,
		CuVehicleTypes &vehicleTypes, int numOfVehicleTypes);
	void createOnGpu(CuEmptyTrips &emptyTrips, int numOfEmptyTrips, CuServiceTrips &serviceTrips, int numOfServiceTrips,
		CuVehicleTypes &vehicleTypes, int numOfVehicleTypes);
	CuMatrix1<KilowattHour> *_emptyTripConsumption;
	CuMatrix1<KilowattHour> *_serviceTripConsumption;
	ConsumptionMatrix *_devicePtr;
};

