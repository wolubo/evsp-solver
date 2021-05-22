#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>

#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/PointInTime.hpp"
#include "ChargingSystem.h"


/// <summary>
/// Repräsentiert einen Fahrzeugtypen.
/// </summary>
class VehicleType
{
	friend class Problem;
public:
	VehicleType() = delete;
	~VehicleType() {}

	/// <summary>
	/// Erzeugt einen neuen Fahrzeugtyp
	/// </summary>
	/// <param name="id"></param>
	/// <param name="theCode"></param>
	/// <param name="theName"></param>
	/// <param name="theVehCost"></param>
	/// <param name="theKmCost"></param>
	/// <param name="theHourCost"></param>
	/// <param name="theBatteryCapacity"></param>
	/// <param name="theConsumptionPerServiceJourneyKm"></param>
	/// <param name="theConsumptionPerDeadheadKm"></param>
	/// <param name="theRechargingCost"></param>
	/// <param name="theRechargingTime"></param>
	/// <param name="supportedChargingSystems"></param>
	VehicleType(const std::string& legacyId, const std::string &theCode, const std::string &theName,
		AmountOfMoney theVehCost, AmountOfMoney theKmCost, AmountOfMoney theHourCost,
		KilowattHour theBatteryCapacity, KilowattHour theConsumptionPerServiceJourneyKm, KilowattHour theConsumptionPerDeadheadKm,
		AmountOfMoney theRechargingCost, DurationInSeconds theRechargingTime,
		//					std::vector<std::shared_ptr<ChargingSystem>> supportedChargingSystems,
		std::string vehCharacteristic, std::string vehClass, std::string curbWeightKg, std::string capacity, std::string slowRechargingTime, std::string fastRechargingTime);

	/// <summary>
	/// Nummer des Fahrzeugtypen
	/// </summary>
	std::string getCode() const;

	/// <summary>
	/// Bezeichnung der Fahrzeugtypen
	/// </summary>
	std::string getName() const;

	/// <summary>
	/// Anschaffungskosten eines Fahrzeugs dieser Fahrzeuggruppe in vollen Geldeinheiten (bspw. Euro).
	/// </summary>
	AmountOfMoney getVehCost() const;

	/// <summary>
	/// Kosten pro Fahrtkilometer in vollen Geldeinheiten (bspw. Euro).
	/// </summary>
	AmountOfMoney getKmCost() const;

	/// <summary>
	/// Kosten des Fahrzeugeinsatzes pro Stunde in vollen Geldeinheiten (bspw. Euro).
	/// </summary>
	AmountOfMoney getHourCost() const;

	/// <summary>
	/// Batteriekapazität in kWh.
	/// </summary>
	KilowattHour getBatteryCapacity() const;

	/// <summary>
	/// Energieverbrauch auf einer Servicefahrt (also eine Fahrt mit Passagieren) in kWh pro km.
	/// </summary>
	KilowattHour getConsumptionPerServiceJourneyKm() const;

	/// <summary>
	/// Energieverbrauch auf einer Verbindungsfahrt (also eine Fahrt ohne Passagiere) in kWh pro km.
	/// </summary>
	KilowattHour getConsumptionPerDeadheadKm() const;

	/// <summary>
	/// Fixkosten für eine Aufladung in vollen Geldbeträgen (bspw. Euro).
	/// Dabei wird nicht zwischen vollständigen Ladungen und Teilladungen unterschieden. Alle Ladevorgänge kosten also dasselbe.
	/// </summary>
	AmountOfMoney getRechargingCost() const;

	/// <summary>
	/// Zeitdauer einer Aufladung in Sekunden.
	/// Dabei wird nicht zwischen vollständigen Ladungen und Teilladungen unterschieden. Alle Ladevorgänge dauern also gleich lang.
	/// </summary>
	DurationInSeconds getRechargingTime() const;

	std::string toString();

	VehicleTypeId getId() const { return _id; }

#ifdef _DEBUG
	void write2file(std::ofstream &txtfile);
	std::string getLegacyId() { return _legacyId; }
#endif

#pragma region private_member_variables
private:
	VehicleTypeId _id;

	std::string _legacyId;

	/// <summary>
	/// Nummer des Fahrzeugtypen
	/// </summary>
	std::string _code;

	/// <summary>
	/// Bezeichnung der Fahrzeugtypen
	/// </summary>
	std::string _name;

	/// <summary>
	/// Anschaffungskosten eines Fahrzeugs dieser Fahrzeuggruppe in vollen Geldeinheiten (bspw. Euro).
	/// </summary>
	AmountOfMoney _vehCost;

	/// <summary>
	/// Kosten pro Fahrtkilometer in vollen Geldeinheiten (bspw. Euro).
	/// </summary>
	AmountOfMoney _kmCost;

	/// <summary>
	/// Kosten des Fahrzeugeinsatzes pro Stunde in vollen Geldeinheiten (bspw. Euro).
	/// </summary>
	AmountOfMoney _hourCost;

	/// <summary>
	/// Batteriekapazität in kWh.
	/// </summary>
	KilowattHour _batteryCapacity;

	/// <summary>
	/// Energieverbrauch auf einer Servicefahrt (also eine Fahrt mit Passagieren) in kWh pro km.
	/// </summary>
	KilowattHour _consumptionPerServiceJourneyKm;

	/// <summary>
	/// Energieverbrauch auf einer Verbindungsfahrt (also eine Fahrt ohne Passagiere) in kWh pro km.
	/// </summary>
	KilowattHour _consumptionPerDeadheadKm;

	/// <summary>
	/// Fixkosten für eine Aufladung in vollen Geldbeträgen (bspw. Euro).
	/// Dabei wird nicht zwischen vollständigen Ladungen und Teilladungen unterschieden. Alle Ladevorgänge kosten also dasselbe.
	/// </summary>
	AmountOfMoney _rechargingCost;

	/// <summary>
	/// Zeitdauer einer Aufladung in Sekunden.
	/// Dabei wird nicht zwischen vollständigen Ladungen und Teilladungen unterschieden. Alle Ladevorgänge dauern also gleich lang.
	/// </summary>
	DurationInSeconds _rechargingTime;


#pragma endregion

#pragma region unsupported_attributes
	// Die folgenden Attribute sind in den Input-Files enthalten, werden aber vom System derzeit (noch) nicht unterstützt.
	std::string _vehCharacteristic;
	std::string _vehClass;
	std::string _curbWeightKg;
	std::string _capacity;
	std::string _slowRechargingTime;
	std::string _fastRechargingTime;
	//				std::vector<std::shared_ptr<ChargingSystem>> _supportedChargingSystems;
#pragma endregion

};
