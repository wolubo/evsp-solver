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
	/// Knoten diesen Typs repr�sentieren die Kombination aus einem bestimmten Fahrzeugtypen und einem bestimmten Depot.
	/// F�r jede m�gliche Kombination aus Depots und Fahrzeugtypen gibt es im Entscheidungsnetz einen solchen Knoten.
	/// Steht f�r das Startdepot eines Fahrzeugtypen. Es handelt sich also um einen Knoten auf der ersten Ebene des 
	/// Netzes, auf der entschieden wird, in welchem Depot und mit welchem Fahrzeug der jeweilige Umlauf startet.
	/// </summary>
	VehTypeDepotNode,

	/// <summary>
	/// Es handelt sich um einen Servicefahrtknoten. Also um einen Knoten, in dem entschieden wird, welche Servicefahrt als 
	/// n�chstes durchgef�hrt wird.
	/// </summary>
	ServiceTripNode,

	/// <summary>
	/// Der Knoten repr�sentiert eine Haltestelle mit Ladestation.
	/// </summary>
	ChargingStationNode,

	/// <summary>
	/// Der Knotentyp ist undefiniert!
	/// </summary>
	INVALID

};

