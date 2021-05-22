#pragma once

#include "EvspLimits.h"
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuVector1.hpp"
#include "CuMatrix1.hpp"
#include "CuVehicleTypes.h"


///<summary>
/// Container für alle Fahrzeugtypgruppen.
///</summary>
class CuVehicleTypeGroups
{
public:
	CuVehicleTypeGroups(int maxNumOfVehicleTypeGroups, int maxNumOfVehicleTypes);
	~CuVehicleTypeGroups();

	CuVehicleTypeGroups* getDevPtr();

	///<summary>
	/// Fügt eine neue (leere) Fahrzeugtypgruppe hinzu.
	///</summary>
	void addGroup();

	///<summary>
	/// Fügt einer existierenden Fahrzeugtypgruppe einen Fahrzeugtypen hinzu.
	///</summary>
	///<param name="groupId"></param>
	///<param name="typeId"></param>
	void addType(VehicleTypeGroupId groupId, VehicleTypeId typeId);

	///<summary>
	/// Liefert die Anzahl der Fahrzeugtypgruppen.
	///</summary>
	///<returns>Anzahl der Fahrzeugtypgruppen.</returns>
	CU_HSTDEV int getNumOfVehicleTypeGroups() const;

	///<summary>
	/// Liefert die Anzahl der Fahrzeugtypen in einer Gruppe.
	///</summary>
	///<param name="groupId">Id der Gruppe, zu der die Anzahl der Fahrzeugtypen geliefert werden soll.</param>
	///<returns>Anzahl der Fahrzeugtypen in der Gruppe "groupId".</returns>
	CU_HSTDEV int getNumOfVehicleTypes(VehicleTypeGroupId groupId) const;

	///<summary>
	/// Prüft, ob ein Fahrzeugtyp in einer bestimmten Gruppe enthalten ist.
	///</summary>
	///<param name="groupId">Id der Gruppe, die geprüft werden soll.</param>
	///<param name="typeId">Id des Fahrzeugtyps.</param>
	///<returns>True, wenn der Fahrzeugtyp "typeId" in der Gruppe "groupId" enthalten ist. Sonst false.</returns>
	CU_HSTDEV bool hasVehicleType(VehicleTypeGroupId groupId, VehicleTypeId typeId) const;

	///<summary>
	/// Liefert die Id eines Fahrzeugtypen aus der Gruppe "groupId".
	///</summary>
	///<param name="groupId">Id der Fahrzeugtypgruppe.</param>
	///<param name="idx">Lfd. Index innerhalb der Fahrzeugtypgruppe.</param>
	///<returns>Fahrzeugtyp-Id oder -1, falls die Gruppe keinen Fahrzeugtypen mit dem Index enthält.</returns>
	CU_HSTDEV VehicleTypeId get(VehicleTypeGroupId groupId, int idx) const;

private:
	CuVehicleTypeGroups() : _numOfVehicleTypeGroups(0), _numOfVehicleTypes(0), _vehicleTypeIds(0), _devicePtr(0) {}
	ushort _numOfVehicleTypeGroups;
	CuVector1<ushort> *_numOfVehicleTypes;
	CuMatrix1<VehicleTypeId> *_vehicleTypeIds;
	CuVehicleTypeGroups *_devicePtr;
};

