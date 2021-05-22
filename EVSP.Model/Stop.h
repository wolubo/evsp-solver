#pragma once

#include <string>
#include <map>
#include <vector>

#include "EVSP.BaseClasses/Typedefs.h"



/// <summary>
/// Repräsentiert einen Haltepunkt. Das kann eine Bushaltestelle oder ein Busdepot sein.
/// </summary>
class Stop
{
	friend class Problem;
public:

	Stop() = delete;
	~Stop() {}

	/// <summary>
	/// Erzeugt einen neuen Haltepunkt
	/// </summary>
	/// <param name="code">Nummer des Haltepunkts</param>
	/// <param name="name">Bezeichnung des Haltepunkts</param>
	/// <param name="vehCapacityForCharging">???</param>
	Stop(const std::string& legacyId, const std::string &code, const std::string &name, bool chargingStation, std::string vehCapacityForCharging);

	/// <summary>
	/// Nummer des Haltepunkts.
	/// </summary>
	std::string getCode() const;

	/// <summary>
	/// Bezeichnung des Haltepunkts.
	/// </summary>
	std::string getName() const;

	/// <summary>
	/// True, wenn Fahrzeuge an dieser Haltestelle aufgeladen werden können.
	/// </summary>
	bool isChargingStation() const;

	void setChargingStation(bool isChargingStation) { _isChargingStation = isChargingStation; }

	/// <summary>
	/// Liefert true, wenn es sich um eine Bushaltestelle handelt. Also um einen Haltepunkt, an dem Servicefahrten 
	/// abgehen und/oder ankommen.
	/// </summary>
	bool isBusStop() const;

	/// <summary>
	/// Liefert true, wenn es sich um ein Busdepot handelt. Also einen Betriebshof, an dem Fahrzeuge außerhalb der Betriebszeiten
	/// abgestellt werden und von dem aus alle Fahrzeugumläufe der hier stationierten Fahrzeuge starten.
	/// False, wenn es sich um eine Bushaltestelle handelt. Also um eine Haltestelle, an der Servicefahrten beginnen und enden.
	/// </summary>
	bool isDepot() const;

	void setDepot(bool isDepot);

	std::string toString();

	StopId getId() const { return _id; }

	void write2file(std::ofstream &txtfile);
	std::string getLegacyId() { return _legacyId; }

#pragma region private_member_variables
private:
	StopId _id;

	/// <summary>
	/// Nummer des Haltepunkts.
	/// </summary>
	std::string _code;

	/// <summary>
	/// Bezeichnung des Haltepunkts.
	/// </summary>
	std::string _name;

	/// <summary>
	/// True, wenn Fahrzeuge an dieser Haltestelle aufgeladen werden können.
	/// </summary>
	bool _isChargingStation;

	/// <summary>
	/// True, wenn es sich um ein Busdepot handelt. Also einen Betriebshof, an dem Fahrzeuge außerhalb der Betriebszeiten
	/// abgestellt werden und von dem aus alle Fahrzeugumläufe der hier stationierten Fahrzeuge starten.
	/// False, wenn es sich um eine Bushaltestelle handelt. Also um eine Haltestelle, an der Servicefahrten beginnen und enden.
	/// </summary>
	bool _isDepot;

	std::string _legacyId;

#pragma endregion

#pragma region unsupported_attributes
	// Die folgenden Attribute sind in den Input-Files enthalten, werden aber vom System derzeit (noch) nicht unterstützt.
	std::string _vehCapacityForCharging;
#pragma endregion
};
