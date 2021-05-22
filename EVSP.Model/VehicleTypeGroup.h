#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>

#include "EVSP.BaseClasses/Typedefs.h"
#include "VehicleType.h"



/// <summary>
/// Repräsentiert eine Fahrzeugtypgruppe.
/// </summary>
class VehicleTypeGroup
{
	friend class Problem;
public:
	VehicleTypeGroup() = delete;
	~VehicleTypeGroup() {}

	/// <summary>
	/// Erzeugt eine neue Fahrzeugtypgruppe
	/// </summary>
	/// <param name="theCode">Nummer der Fahrzeugtypgruppe</param>
	/// <param name="theName">Bezeichnung der Fahrzeugtypgruppe</param>
	VehicleTypeGroup(const std::string& legacyId, const std::string &theCode, const std::string &theName);

	void addVehicleType(std::shared_ptr<VehicleType> vehicleType);

	/// <summary>
	/// Nummer der Fahrzeugtypgruppe
	/// </summary>
	std::string getCode() const;

	/// <summary>
	/// Bezeichnung der Fahrzeugtypgruppe
	/// </summary>
	std::string getName() const;

	/**
	* Liefert die Anzahl der Fahrzeugtypen in der Gruppe.
	*/
	int getNumberOfVehicleTypes() const;

	/**
	* Liefert einen Fahrzeugtypen aus der Gruppe.
	* @param index
	*/
	std::shared_ptr<VehicleType> getVehicleType(VehicleTypeId id) const;

	std::string toString();

	VehicleTypeGroupId getId() const { return _id; }

	void write2file(std::ofstream &txtfile);
	//void vehType2vehTypeGrp(std::vector<std::string> &result, std::vector<VehicleType> vehTypes);
	std::string getLegacyId() { return _legacyId; }

#pragma region private_member_variables
private:
	VehicleTypeGroupId _id;

	/// <summary>
	/// Nummer der Fahrzeugtypgruppe
	/// </summary>
	std::string _code;

	/// <summary>
	/// Bezeichnung der Fahrzeugtypgruppe
	/// </summary>
	std::string _name;

	/// <summary>
	/// Liste der Fahrzeugtypen, die zu dieser Fahrzeugtypgruppe gehören.
	/// </summary>
	std::vector<std::shared_ptr<VehicleType>> _vehicleTypes;

	std::string _legacyId;

#pragma endregion

};
