#pragma once

#include "EmptyTrip.h"
#include "VehicleType.h"
#include "VehicleTypeGroup.h"
#include "BusRoute.h"
#include "ServiceTrip.h"
#include "Stop.h"

using namespace std;


class Problem {
public:
	Problem();
	~Problem();

	void addRoute(shared_ptr<BusRoute> newRoute);

	void addServiceTrip(shared_ptr<ServiceTrip> newServiceTrip);

	void addStop(shared_ptr<Stop> newStop);

	void addEmptyTrip(shared_ptr<EmptyTrip> newEmptyTrip);

	void addVehicleType(shared_ptr<VehicleType> newVehicleType);

	void addVehicleTypeGroup(shared_ptr<VehicleTypeGroup> newVehicleTypeGroup);

	const vector<shared_ptr<BusRoute>>& getRoutes() const;

	const vector<shared_ptr<ServiceTrip>>& getServiceTrips() const;

	const vector<shared_ptr<Stop>>& getStops() const;

	vector<shared_ptr<Stop>> getBusStops() const;

	vector<shared_ptr<Stop>> getDepots() const;

	vector<shared_ptr<Stop>> getChargingStations() const;

	const vector<shared_ptr<EmptyTrip>>& getEmptyTrips() const;

	const vector<shared_ptr<VehicleType>>& getVehicleTypes() const;

	const vector<shared_ptr<VehicleTypeGroup>>& getVehicleTypeGroups() const;

	DistanceInMeters getAvgEmptyTripDistance();
	DurationInSeconds getAvgEmptyTripDuration();
	DistanceInMeters getAvgServiceTripDistance();
	DurationInSeconds getAvgServiceTripDuration();

	/// <summary>
	/// Lege nach dem Zufallsprinzip fest, an welchen Haltestellen aufgeladen werden kann.
	/// Gibt es keine Ausrück- oder keine Einrückfahrt, so erzeuge eine entsprechende Verbindungsfahrt mit Durchschnittswerten.
	/// </summary>
	/// <param name="ratio">
	/// Legt den Anteil der Ladestationen an den Bushaltestelle fest (0-100). 
	/// 0    = keine Ladestation (ausser den Depots)
	/// 1-99 = prozentualer Anteil
	/// 100  = alle Bushaltestellen sind Ladestationen.
	/// </param>
	void randomizeChargingStations(int ratio);

	/// <summary>
	/// Das Problem soll als konventionelles Vehicle Scheduling Problem für Fahrzeuge mit Verbrennungsmotoren gelöst werden.
	/// Entferne alle Ladestationen und setze die Batteriekapazitäten der Fahrzeuge auf den höchstmöglichen Wert.
	/// </summary>
	void solveForCombustionVehicles();

	/// <summary>
	/// Überprüfe alle Servicefahrten. 
	/// Gibt es keine Ausrück- oder keine Einrückfahrt, so erzeuge eine entsprechende Verbindungsfahrt mit Durchschnittswerten.
	/// </summary>
	void checkServiceTrips(bool verbose);

	/// <summary>
	/// Überprüfe alle Verbindungsfahrten. 
	/// Ist die Entfernung nicht plausibel (_distance==0): Ersetze sie durch die durchschnittliche Entfernung aller Verbindungsfahrten.
	/// Ist die Dauer nicht plausibel (_duration==0): Ersetze sie durch die durchschnittliche Dauer aller Verbindungsfahrten.
	/// </summary>
	void checkEmptyTrips();

	void printStatistic();

private:
	void computeEmptyTripAvg();
	void computeServiceTripAvg();

	vector<shared_ptr<EmptyTrip>> _emptyTrips;
	vector<shared_ptr<VehicleType>> _vehicleTypes;
	vector<shared_ptr<VehicleTypeGroup>> _vehicleTypeGroups;
	vector<shared_ptr<BusRoute>> _routes;
	vector<shared_ptr<ServiceTrip>> _serviceTrips;
	vector<shared_ptr<Stop>> _stops;

	DistanceInMeters _avgEmptyTripDistance;
	DurationInSeconds _avgEmptyTripDuration;
	DistanceInMeters _avgServiceTripDistance;
	DurationInSeconds _avgServiceTripDuration;
};

