#pragma once

#include <memory>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuMatrix1.hpp"
#include "CuServiceTrips.h"
#include "CuVehicleTypes.h"
#include "CuVehicleTypeGroups.h"

using namespace std;

class CuProblem;

/// <summary>
/// Enth�lt zu jeder zul�ssigen Kombination aus ServiceTrip und VehicleType sowohl die zeit- als auch die streckenabh�ngigen Kosten.
/// Enth�lt ausserdem den Batterieverbrauch f�r jede zul�ssige Kombination aus ServiceTrip und VehicleType.
/// </summary>
class ServiceTripCostMatrix
{
public:
	ServiceTripCostMatrix(CuServiceTrips *serviceTrips, CuVehicleTypes *vehicleTypes,
		CuVehicleTypeGroups *vehicleTypeGroups, PlattformConfig plattform);
	~ServiceTripCostMatrix();

	ServiceTripCostMatrix* getDevPtr();
	void copyToHost();

	/// <summary>
	/// Liefert die streckenabh�ngigen Kosten der Servicefahrt.
	/// </summary>
	CU_HSTDEV AmountOfMoney getDistanceDependentCosts(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Liefert die zeitnabh�ngigen Kosten der Servicefahrt.
	/// </summary>
	CU_HSTDEV AmountOfMoney getTimeDependentCosts(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Liefert die Gesamtkosten der Servicefahrt.
	/// </summary>
	CU_HSTDEV AmountOfMoney getTotalCost(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Liefert den Batterieverbrauch der Servicefahrt.
	/// </summary>
	CU_HSTDEV KilowattHour getConsumption(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Liefert TRUE, wenn die Servicefahrt durch den Fahrzeugtypen bedient werden kann.
	/// </summary>
	CU_HSTDEV bool combinationIsValid(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Liefert den Fahrzeugtypen, mit dem sich die Servicefahrt zu den geringsten Kosten durchf�hren l�sst.
	/// </summary>
	CU_HSTDEV VehicleTypeId getBestVehicleType(ServiceTripId serviceTripId) const;

private:
	void createCpu(CuServiceTrips *serviceTrips, int numOfServiceTrips, CuVehicleTypes *vehicleTypes, int numOfVehicleTypes, 
		CuVehicleTypeGroups *vehicleTypeGroups);
	void createGpu(CuServiceTrips *serviceTrips, int numOfServiceTrips, CuVehicleTypes *vehicleTypes, int numOfVehicleTypes,
		CuVehicleTypeGroups *vehicleTypeGroups);
	void startCreateKernel(int numOfServiceTrips, int numOfVehicleTypes, CuServiceTrips *serviceTrips,
		CuVehicleTypes *vehicleTypes, CuVehicleTypeGroups *vehicleTypeGroups);
	void startFindBestVehTypeKernel(int numOfServiceTrips, int numOfVehicleTypes);

	CuMatrix1<AmountOfMoney> *_distanceDependentCosts;
	CuMatrix1<AmountOfMoney> *_timeDependentCosts;
	CuMatrix1<KilowattHour> *_consumption;
	CuVector1<VehicleTypeId> *_bestVehicleType;
	ServiceTripCostMatrix *_devicePtr;
};

