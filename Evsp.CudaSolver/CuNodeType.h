#pragma once


/// <summary>
/// 
/// </summary>
enum CuNodeType {

	/// <summary>
	/// Es handelt sich um den Wurzelknoten des Netzes. 
	/// </summary>
	RootNode,

	/// <summary>
	/// Knoten diesen Typs repräsentieren die Kombination aus einem bestimmten Fahrzeugtypen und einem bestimmten Depot.
	/// Für jede mögliche Kombination aus Depots und Fahrzeugtypen gibt es im Entscheidungsnetz einen solchen Knoten.
	/// Steht für das Startdepot eines Fahrzeugtypen. Es handelt sich also um einen Knoten auf der ersten Ebene des 
	/// Netzes, auf der entschieden wird, in welchem Depot und mit welchem Fahrzeug der jeweilige Umlauf startet.
	/// </summary>
	VehTypeDepotNode,

	/// <summary>
	/// Es handelt sich um einen Servicefahrtknoten. Also um einen Knoten, in dem entschieden wird, welche Servicefahrt als 
	/// nächstes durchgeführt wird.
	/// </summary>
	ServiceTripNode,

	/// <summary>
	/// Der Knoten repräsentiert eine Haltestelle mit Ladestation.
	/// </summary>
	ChargingStationNode,

	/// <summary>
	/// Der Knotentyp ist undefiniert!
	/// </summary>
	INVALID

};

