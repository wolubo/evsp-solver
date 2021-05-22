#pragma once


///<summary>
/// Maximale Anzahl der Servicefahrten.
///</summary>
const int Max_ServiceTrips = 10710; // __int16


///<summary>
/// Maximale Anzahl der Verbindungsfahrten.
///</summary>
const int Max_EmptyTrips = 10100; // __int16


///<summary>
/// Maximale Anzahl der Haltestellen.
///</summary>
const int Max_Stops = 207; // unsigned __int8


///<summary>
/// Maximale Anzahl der Depots.
///</summary>
const int Max_Depots = 70; // unsigned __int8


///<summary>
/// Maximale Anzahl der Ladestationen.
///</summary>
const int Max_ChargingStations = 207; // unsigned __int8


///<summary>
/// Maximale Anzahl der Buslinien.
///</summary>
const int Max_Routes = 74; // unsigned __int8


///<summary>
/// Maximale Anzahl der Fahrzeugtypen.
///</summary>
const int Max_VehicleTypes = 9; // unsigned __int8


///<summary>
/// Maximale Anzahl der Fahrzeugtypgruppen.
///</summary>
const int Max_VehicleTypeGroups = 9; // unsigned __int8


///<summary>
/// Maximale Anzahl der Segmente pro Umlauf.
///</summary>
const int Max_NumOfCirculationSegments = 40;

const int Max_NumOfCirculations = 600;

///<summary>
/// Maximale Anzahl der Knoten in einem CuACO-Entscheidungsnetz.
///</summary>
const int Max_Nodes = Max_ServiceTrips + Max_Depots + Max_VehicleTypes + Max_ChargingStations + 2; // +2 für Rootknoten und Endknoten

///</summary>
/// Maximale Anzahl ausgehender Kanten pro Knoten in einem CuACO-Entscheidungsnetz.
///</summary>
const int Max_EdgesPerNode = 11000;
