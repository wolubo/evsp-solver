#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EvspLimits.h"
#include "CuNodeType.h"
#include "EVSP.BaseClasses/Typedefs.h"


/// <summary>
/// Verwaltet die Knoten des ÖPNV-Netzes für die GPU. Die Knoten sind logisch in einem Vektor angeordnet. Der Index 
/// entspricht der Id des Knotens.
/// </summary>
class CuNodes {
public:

	void init();

	/// <summary>
	/// Liefert die aktuelle Anzahl der Knoten.
	/// </summary>
	/// <param name="nodeType"></param>
	/// <param name="payloadId"></param>
	/// <returns>Knotenanzahl</returns>
	CU_HSTDEV ushort getNumOfNodes();

	/// <summary>
	/// Fügt einen neuen Knoten hinzu.
	/// </summary>
	/// <param name="nodeType"></param>
	/// <param name="payloadId"></param>
	/// <param name="vehTypeId"></param>
	/// <returns>Id des neuen Knotens</returns>
	CU_HSTDEV NodeId addNode(CuNodeType nodeType, short payloadId, VehicleTypeId vehTypeId);

	/// <summary>
	/// Liefert den Knotentypen eines Knotens.
	/// </summary>
	/// <param name="nodeId">Id des Knotens</param>
	/// <returns>Knotentyp</returns>
	CU_HSTDEV CuNodeType getNodeType(NodeId nodeId);

	/// <summary>
	/// Liefert die mit dem Knoten assoziierte Id.
	/// Bedeutung ist abhängig vom nodeType :
	/// RootNode --> keine Bedeutung
	///	DepotNode --> Haltestellen-Id des Depots(CuStops)
	/// ServiceTripNode --> Id einer Servicefahrt(CuServiceTrips)
	/// VehicleTypeNode --> Id eines Fahrzeugtypen(CuVehicleType)
	/// ChargingStationNode --> Haltestellen-Id der Ladestation (CuStops)
	/// </summary>
	/// <param name="nodeId">Id des Knotens</param>
	/// <returns></returns>
	CU_HSTDEV short getPayloadId(NodeId nodeId);

	/// <summary>
	/// Liefert die mit dem Knoten assoziierte Fahrzeugtyp-Id.
	/// </summary>
	/// <param name="nodeId">Id des Knotens</param>
	/// <returns>Fahrzeugtyp-Id</returns>
	CU_HSTDEV VehicleTypeId getVehTypeId(NodeId nodeId);

private:

	/// <summary>
	/// Aktuelle Knotenanzahl.
	/// </summary>
	ushort _numOfNodes;

	/// <summary>
	/// Definiert den Knotentyp.
	/// </summary>
	CuNodeType _nodeType[Max_Nodes];

	/// <summary>
	/// Bedeutung ist abhängig vom nodeType :
	/// RootNode--> keine Bedeutung
	/// DepotNode--> Haltestellen - Id eines Depots(CuStops)
	/// ServiceTripNode--> Id einer Servicefahrt(CuServiceTrips)
	/// VehicleTypeNode--> Id eines Fahrzeugtypen(CuVehicleType)
	/// </summary>
	short _payloadId[Max_Nodes];

	VehicleTypeId _vehTypeId[Max_Nodes];
};

