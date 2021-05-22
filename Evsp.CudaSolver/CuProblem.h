#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "EvspLimits.h"
#include "CuMatrix2.hpp"
#include "CuServiceTrips.h"
#include "CuEmptyTrips.h"
#include "CuStops.h"
#include "CuVehicleTypes.h"
#include "CuVehicleTypeGroups.h"
#include "ConsumptionMatrix.h"
#include "ConnectionMatrix.h"
#include "PrevNextMatrix.h"
#include "ServiceTripCostMatrix.h"
#include "EmptyTripCostMatrix.h"
#include "ChargingMatrix.h"

/// <summary>
/// Diese Klasse dient als Container für alle Objekte, die zur Lösung eines EVSP durch einen Cuda-Solver nötig sind.
/// </summary>
class CuProblem
{

public:
	CuProblem(CuStops *stops, CuEmptyTrips *emptyTrips, CuServiceTrips *serviceTrips, CuVehicleTypes *vehicleTypes, CuVehicleTypeGroups *vehicleTypeGroups, PlattformConfig plattform, bool performChecks);
	~CuProblem();

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory) und speichert die Adresse der Kopie 
	/// im Attribut 'deviceObject'. Alle weiteren Aufrufe liefern einen Pointer auf diese Kopie (also den Inhalt von 'deviceObject').
	/// </summary>
	CuProblem* getDevPtr();

	void copyToHost();

	CU_HSTDEV  CuStops& getStops() const { return *_stops; }
	CU_HSTDEV  CuEmptyTrips& getEmptyTrips() const { return *_emptyTrips; }
	CU_HSTDEV  CuServiceTrips& getServiceTrips() const { return *_serviceTrips; }
	CU_HSTDEV  CuVehicleTypes& getVehicleTypes() const { return *_vehicleTypes; }
	CU_HSTDEV  CuVehicleTypeGroups& getVehicleTypeGroups() const { return *_vehicleTypeGroups; }

	CU_HSTDEV  ConnectionMatrix& getConnectionMatrix() const { return *_connectionMatrix; }
	CU_HSTDEV  PrevNextMatrix& getPrevNextMatrix() const { return *_prevNextMatrix; }
	CU_HSTDEV  ServiceTripCostMatrix& getServiceTripCostMatrix() const { return *_serviceTripCost; }
	CU_HSTDEV  EmptyTripCostMatrix& getEmptyTripCostMatrix() const { return *_emptyTripCost; }
	CU_HSTDEV  ChargingMatrix& getChargingMatrix() const { return *_chargingMatrix; }

	/// <summary>
	/// Liefert die streckenabhängigen Kosten, die durch eine Verbindungsfahrt entstehen.
	/// </summary>
	/// <param name="departure">Id der Starthaltestelle</param>
	/// <param name="destination">Id der Zielhaltestelle</param>
	/// <param name="vehTypeId">Id des Fahrzeugtyps</param>
	/// <returns></returns>
	CU_HSTDEV AmountOfMoney getEmptyTripCosts(StopId departure, StopId destination, VehicleTypeId vehTypeId) const;

	/// <summary>
	/// Liefert die streckenabhängigen Kosten, die durch eine Servicefahrt entstehen.
	/// Das Ergebnis beinhaltet auch die Kosten durch ggf. nötige Verbindungsfahrten.
	/// </summary>
	/// <param name="departure">Id der Starthaltestelle</param>
	/// <param name="destination">Id der Zielhaltestelle</param>
	/// <param name="servTripId">Id der Servicefahrt</param>
	/// <param name="servTrip">Daten der Servicefahrt</param>
	/// <param name="vehTypeId">Id des Fahrzeugtyps</param>
	/// <returns></returns>
	CU_HSTDEV AmountOfMoney getServiceTripCosts(StopId departure, StopId destination, ServiceTripId servTripId, const CuServiceTrip &servTrip, VehicleTypeId vehTypeId) const;

	/// <summary>
	/// Liefert die streckenabhängigen Kosten, die durch eine Servicefahrt entstehen.
	/// Das Ergebnis beinhaltet auch die Kosten durch ggf. nötige Verbindungsfahrten.
	/// </summary>
	/// <param name="departure">Id der Starthaltestelle</param>
	/// <param name="destination">Id der Zielhaltestelle</param>
	/// <param name="servTripId">Id der Servicefahrt</param>
	/// <param name="vehTypeId">Id des Fahrzeugtyps</param>
	/// <returns></returns>
	CU_HSTDEV AmountOfMoney getServiceTripCosts(StopId departure, StopId destination, ServiceTripId servTripId, VehicleTypeId vehTypeId) const;

	/// <summary>
	/// Ermittelt den Energieverbrauch für das Absolvieren einer Verbindungsfahrt.
	/// </summary>
	/// <param name="departure">Id der Starthaltestelle</param>
	/// <param name="destination">Id der Zielhaltestelle</param>
	/// <param name="vehTypeId"> Id des Fahrzeugtyps, mit dem die Verbindungsfahrt durchgeführt werden soll.</param>
	CU_HSTDEV KilowattHour getEmptyTripConsumption(StopId departure, StopId destination, VehicleTypeId vehTypeId) const;

	/// <summary>
	/// Ermittelt den Energieverbrauch für das Absolvieren einer Servicefahrt. Dabei werden ggf. 
	/// nötige Verbindungsfahrten berücksichtigt.
	/// </summary>
	/// <param name="departure">Id der Starthaltestelle</param>
	/// <param name="destination">Id der Zielhaltestelle</param>
	/// <param name="servTripId">Id der Servicefahrt</param>
	/// <param name="servTrip">Daten der zu prüfenden Servicefahrt</param>
	/// <param name="vehTypeId"> Id des Fahrzeugtyps, mit dem die Servicefahrt durchgeführt werden soll.</param>
	CU_HSTDEV KilowattHour getServiceTripConsumption(StopId departure, StopId destination, ServiceTripId servTripId, const CuServiceTrip &servTrip, VehicleTypeId vehTypeId) const;

	/// <summary>
	/// Ermittelt den Energieverbrauch für das Absolvieren einer Servicefahrt. Dabei werden ggf. 
	/// nötige Verbindungsfahrten berücksichtigt.
	/// </summary>
	/// <param name="departure">Id der Starthaltestelle</param>
	/// <param name="destination">Id der Zielhaltestelle</param>
	/// <param name="servTripId">Id der Servicefahrt</param>
	/// <param name="vehTypeId"> Id des Fahrzeugtyps, mit dem die Servicefahrt durchgeführt werden soll.</param>
	CU_HSTDEV KilowattHour getServiceTripConsumption(StopId departure, StopId destination, ServiceTripId servTripId, VehicleTypeId vehTypeId) const;

private:
	CuProblem() : _stops(0), _emptyTrips(0), _serviceTrips(0), _vehicleTypes(0), _vehicleTypeGroups(0), _connectionMatrix(0), _prevNextMatrix(0), _chargingMatrix(0), 
		_serviceTripCost(0), _emptyTripCost(0), _deviceObject(0) {}
	CuStops *_stops;
	CuEmptyTrips *_emptyTrips;
	CuServiceTrips *_serviceTrips;
	CuVehicleTypes *_vehicleTypes;
	CuVehicleTypeGroups *_vehicleTypeGroups;
	ConnectionMatrix *_connectionMatrix;
	PrevNextMatrix *_prevNextMatrix;
	ChargingMatrix* _chargingMatrix;
	ServiceTripCostMatrix *_serviceTripCost;
	EmptyTripCostMatrix *_emptyTripCost;
	CuProblem *_deviceObject;
};

