#pragma once

#include <memory>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuMatrix1.hpp"
#include "CuEmptyTrips.h"
#include "CuVehicleTypes.h"


using namespace std;

class CuProblem;

/// <summary>
/// Enthält zu jeder zulässigen Kombination aus EmptyTrip und VehicleType sowohl die zeit- als auch die streckenabhängigen Kosten.
/// Enthält ausserdem den Batterieverbrauch für jede zulässige Kombination aus EmptyTrip und VehicleType.
/// </summary>
class EmptyTripCostMatrix
{
public:
	EmptyTripCostMatrix(CuEmptyTrips *emptyTrips, CuVehicleTypes *vehicleTypes, PlattformConfig plattform);
	~EmptyTripCostMatrix();

	EmptyTripCostMatrix* getDevPtr();
	void copyToHost();

	/// <summary>
	/// Liefert die streckenabhängigen Kosten der Verbindungsfahrt.
	/// </summary>
	CU_HSTDEV AmountOfMoney getDistanceDependentCosts(EmptyTripId emptyTripId, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Liefert die zeitnabhängigen Kosten der Verbindungsfahrt.
	/// </summary>
	CU_HSTDEV AmountOfMoney getTimeDependentCosts(EmptyTripId emptyTripId, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Liefert die Gesamtkosten der Verbindungsfahrt.
	/// </summary>
	CU_HSTDEV AmountOfMoney getTotalCost(EmptyTripId emptyTripId, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Liefert den Batterieverbrauch der Verbindungsfahrt.
	/// </summary>
	CU_HSTDEV KilowattHour getConsumption(EmptyTripId emptyTripId, VehicleTypeId vehicleTypeId) const;

private:
	void createCpu(CuEmptyTrips *emptyTrips, int numOfEmptyTrips, CuVehicleTypes *vehicleTypes, int numOfVehicleTypes);
	void createGpu(CuEmptyTrips *emptyTrips, int numOfEmptyTrips, CuVehicleTypes *vehicleTypes, int numOfVehicleTypes);

	CuMatrix1<AmountOfMoney> *_distanceDependentCosts;
	CuMatrix1<AmountOfMoney> *_timeDependentCosts;
	CuMatrix1<KilowattHour> *_consumption;

	EmptyTripCostMatrix *_devicePtr;
};

