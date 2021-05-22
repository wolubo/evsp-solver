#pragma once

#include "EVSP.BaseClasses\DeviceLaunchParameters.h"
#include "CuVehicleTypes.h"
#include "MatrixCreator.h"


class AntsRaceState {
public:
	CU_HSTDEV AntsRaceState(CuProblem *problem, ConsumptionMatrix *consumption, float chargeLimit, bool verbose);
	
	CU_HSTDEV AntsRaceState(const AntsRaceState& other);
	
	CU_HSTDEV ~AntsRaceState();

	/// <summary>
	/// Beginnt den nächsten Umlauf ("Fahrer steigt im Depot in ein konkretes Fahrzeug ein").
	/// Die Ausrückfahrt ist Bestandteil der ersten Servicefahrt.
	/// </summary>
	CU_HSTDEV void startCirculation(StopId startDepotId, VehicleTypeId vehTypeId);

	/// <summary>
	/// Beendet den aktuellen Umlauf (der Fahrer fährt zurück ins Depot und lädt das Fahrzeug auf).
	/// Beinhaltet also auch die Einrückfahrt.
	/// </summary>
	CU_HSTDEV void endCirculation();

	/// <summary>
	/// Prüft, ob eine bestimmte Servicefahrt durchgeführt werden kann.
	/// </summary>
	CU_HSTDEV bool canProcessServiceTrip(ServiceTripId servTripId);

	/// <summary>
	/// Führt eine Servicefahrt durch.
	/// Berechnet den Verbrauch und die streckenabhängigen Kosten einer Servicefahrt und aktualisiert die entsprechenden Variablen.
	/// Aktualisiert die aktuelle Zeit.
	/// Beinhaltet auch eine ggf. nötige Verbindungsfahrt zur Starthaltestelle der Servicefahrt.
	/// </summary>
	CU_HSTDEV void processServiceTrip(ServiceTripId servTripId);

	/// <summary>
	/// Führt eine Aufladung durch.
	/// Beinhaltet auch eine ggf. nötige Verbindungsfahrt zur Ladestation.
	/// </summary>
	CU_HSTDEV void processCharging(EmptyTripId emptyTripId);
	CU_HSTDEV void processCharging(StopId chargingStationId);

	/// <summary>
	/// Prüft, ob das Aufladen sinnvoll ist.
	/// </summary>
	CU_HSTDEV bool canProcessCharging(StopId chargingStationId);

	CU_HSTDEV PointInTime getCurrentTime() { return _currentTime; }

	CU_HSTDEV StopId getCurrentStopId() { return _currentStopId; }

	/// <summary>
	/// Führt eine Verbindungsfahrt durch.
	/// Berechnet den Verbrauch und die streckenabhängigen Kosten einer Verbindungsfahrt und aktualisiert die entsprechenden 
	/// Variablen. Aktualisiert die aktuelle Zeit.
	/// <returns>Dauer der Fahrt in Sekunden.</returns>
	/// </summary>
	CU_HSTDEV DurationInSeconds processEmptyTrip(EmptyTripId emptyTripId);
	CU_HSTDEV DurationInSeconds processEmptyTrip(StopId toStop);

	CU_HSTDEV bool canProcessEmptyTrip(EmptyTripId emptyTripId);
	CU_HSTDEV bool canProcessEmptyTrip(StopId toStop);

	CU_HSTDEV void updateRemainingBatteryCapacity(KilowattHour consumption);

	CU_HSTDEV StopId getStartDepotId() const { return _startDepotId; }
	CU_HSTDEV VehicleTypeId getVehTypeId() const { return _vehTypeId; }
	//CU_HSTDEV KilowattHour getBatteryCapacity() const { return _batteryCapacity; }
	CU_HSTDEV KilowattHour getRemainingBatteryCapacity() const { return _remainingBatteryCapacity; }

private:
	bool _verbose;
	CuProblem *_problem;
	ConsumptionMatrix *_consumption;

	VehicleTypeId _vehTypeId; // Fahrzeugtyp, mit dem der gerade aktuelle Umlauf durchgeführt wird.
	PointInTime _startTime; // Zeitpunkt, zu dem der aktuelle Umlauf startete.
	StopId _startDepotId; // Id des Depots, in dem der aktuelle Umlauf begonnen hat.

	bool _canCharge; // True, wenn die Ladeschwelle unterschritten wurde und das Anfahren von Ladestationen deshalb erlaubt ist.
	//bool _forceCharge; // True, wenn die Mindestbatteriekapazität unterschritten wurde und aufgeladen werden muss.
	float _chargeLimitInPercent;
	KilowattHour _chargeLimit; // Kapazität der Fahrzeugbatterie, ab der das Anfahren von Ladestationen erlaubt ist.
	KilowattHour _remainingBatteryCapacity; // Aktuelle Restkapazität der Fahrzeugbatterie.

	StopId _currentStopId; // Aktuelle Position des Fahrzeugs (Id der Haltestelle).
	PointInTime _currentTime;  // Aktueller Zeitpunkt.
};

