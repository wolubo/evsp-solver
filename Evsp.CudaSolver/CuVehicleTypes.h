#pragma once

//#include "cuda_runtime.h"
#include "CuVector1.hpp"
#include "EvspLimits.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"


struct /*__align__(16)*/ CuVehicleType {

	///<summary>
	/// Anschaffungskosten eines Fahrzeugs dieser Fahrzeuggruppe in vollen Geldeinheiten (bspw. Euro).
	///</summary>
	AmountOfMoney vehCost;

	///<summary>
	/// Kosten pro Fahrtkilometer in vollen Geldeinheiten (bspw. Euro).
	///</summary>
	AmountOfMoney kmCost;

	///<summary>
	/// Kosten des Fahrzeugeinsatzes pro Stunde in vollen Geldeinheiten (bspw. Euro).
	///</summary>
	AmountOfMoney hourCost;

	///<summary>
	/// Batteriekapazität in kWh.
	///</summary>
	KilowattHour batteryCapacity;

	///<summary>
	/// Energieverbrauch auf einer Servicefahrt (also eine Fahrt mit Passagieren) in kWh pro km.
	///</summary>
	KilowattHour consumptionPerServiceJourneyKm;

	///<summary>
	/// Energieverbrauch auf einer Verbindungsfahrt (also eine Fahrt ohne Passagiere) in kWh pro km.
	///</summary>
	KilowattHour consumptionPerDeadheadKm;

	///<summary>
	/// Fixkosten für eine Aufladung in vollen Geldbeträgen (bspw. Euro).
	/// Dabei wird nicht zwischen vollständigen Ladungen und Teilladungen unterschieden. Alle Ladevorgänge kosten also dasselbe.
	///</summary>
	AmountOfMoney rechargingCost;

	///<summary>
	/// Zeitdauer einer Aufladung in Sekunden.
	/// Dabei wird nicht zwischen vollständigen Ladungen und Teilladungen unterschieden. Alle Ladevorgänge dauern also gleich lang.
	///</summary>
	DurationInSeconds rechargingTime;

	CU_HSTDEV AmountOfMoney getTimeDependentCosts(DurationInSeconds duration) const
	{
		int tempCost = (int)((float)duration * ((float)hourCost / 3600.0f));
		return AmountOfMoney(tempCost);
	}

	CU_HSTDEV AmountOfMoney getDistanceDependentCosts(DistanceInMeters distance) const
	{
		int tempCost = (int)((float)distance * ((float)kmCost / 1000.0f));
		return AmountOfMoney(tempCost);
	}

	CU_HSTDEV AmountOfMoney getTotalCost(DistanceInMeters distance, DurationInSeconds duration) const
	{
		return getDistanceDependentCosts(distance) + getTimeDependentCosts(duration);
	}

	CU_HSTDEV KilowattHour getEmptyTripConsumption(DistanceInMeters distance) const
	{
		float tempConsumption = (float)distance * ((float)consumptionPerDeadheadKm / 1000.0f);
		return KilowattHour(tempConsumption);
	}

	CU_HSTDEV KilowattHour getServiceTripConsumption(DistanceInMeters distance) const
	{
		float tempConsumption = (float)distance * ((float)consumptionPerServiceJourneyKm / 1000.0f);
		return KilowattHour(tempConsumption);
	}
};


///<summary>
/// Container für alle Fahrzeugtypen.
///</summary>
class CuVehicleTypes
{
public:
	CuVehicleTypes(int maxNumOfVehicleTypes);
	~CuVehicleTypes();

	CuVehicleTypes* getDevPtr();


	///<summary>
	/// Fügt einen neuen Fahrzeugtypen hinzu.
	///</summary>
	void add(AmountOfMoney vehCost, AmountOfMoney kmCost, AmountOfMoney hourCost, KilowattHour batteryCapacity, KilowattHour consumptionPerServiceJourneyKm, KilowattHour consumptionPerDeadheadKm, AmountOfMoney rechargingCost, DurationInSeconds rechargingTime);


	///<summary>
	/// Liefert die Anzahl der Fahrzeugtypen.
	///</summary>
	///<returns>Maximale Anzahl der Fahrzeugtypen.</returns>
	CU_HSTDEV int getNumOfVehicleTypes() const;

	///<summary>
	/// 
	///</summary>
	///<param name='id'>Id des Fahrzeugtypen.</param>
	///<returns></returns>
	CU_HSTDEV CuVehicleType& getVehicleType(VehicleTypeId vehicleTypeId) const;

private:
	CuVehicleTypes() {
		_numOfVehicleTypes = 0;
		_vehicleTypes = 0;
		_devicePtr = 0;
	}
	ushort _numOfVehicleTypes;
	CuVector1<CuVehicleType> *_vehicleTypes;
	CuVehicleTypes *_devicePtr;
};

