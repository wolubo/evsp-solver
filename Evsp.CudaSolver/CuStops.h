#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "CuVector1.hpp"
#include "EvspLimits.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuBoolVector1.hpp"

using namespace std;


struct /*__align__(2)*/ CuStop {

	///<summary>
	/// Handelt es sich um ein Depot? Andernfalls ist es eine normale Bushaltestelle.
	///</summary>
	bool isDepot;

	///<summary>
	/// Verfügt die Haltestelle über eine Ladestation?
	///</summary>
	bool isChargingStation;
};


///<summary>
/// Container für alle Haltestellen (Bushaltestellen, Depots und Ladestationen). 
/// Optimiert zur Verwendung in Cuda-Kerneln.
///</summary>
class CuStops
{
public:
	CuStops(int maxNumOfStops, int maxNumOfDepots, int maxNumOfChargingStations);
	~CuStops();

	CuStops* getDevPtr();

	void add(bool depot, bool charging);

	///<summary>
	/// Liefert die Anzahl der Haltestellen (darin sind auch Depots und Ladestationen enthalten).
	///</summary>
	///<returns>Anzahl der Haltestellen.</returns>
	CU_HSTDEV int getNumOfStops() const;

	///<summary>
	/// Liefert die Anzahl der aktuell hinterlegten Depots.
	///</summary>
	///<returns>Anzahl der Depots.</returns>
	CU_HSTDEV int getNumOfDepots() const;

	///<summary>
	/// Liefert die Anzahl der aktuell hinterlegten Ladestationen.
	///</summary>
	///<returns>Anzahl der Ladestationen.</returns>
	CU_HSTDEV int getNumOfChargingStations() const;

	///<summary>
	/// Liefert die Id der letzten Haltestelle im Container.
	///</summary>
	///<returns>Haltestellen-Id</returns>
	CU_HSTDEV StopId lastStopId() const {
		if (_numOfStops > 0)
			return StopId(_numOfStops - 1);
		else
			return StopId::invalid();
	}

	CU_HSTDEV ChargingStationId lastChargingStationId() const {
		if (_numOfChargingStations > 0)
			return ChargingStationId(_numOfChargingStations - 1);
		else
			return ChargingStationId::invalid();
	}

	CU_HSTDEV DepotId lastDepotId() const {
		if (_numOfDepots > 0)
			return DepotId(_numOfDepots - 1);
		else
			return DepotId::invalid();
	}

	///<summary>
	/// 
	///</summary>
	///<param name="stopId">Haltestellen-Id</param>
	///<returns></returns>
	CU_HSTDEV const CuStop& getStop(StopId stopId) const;

	///<summary>
	/// Liefert die Haltestellen-Id eines Depots.
	///</summary>
	///<param name='depotId'>Id des Depots.</param>
	///<returns>Haltestellen-Id des Depots</returns>
	CU_HSTDEV StopId getStopIdOfDepot(DepotId depotId) const;

	///<summary>
	/// Liefert die Haltestellen-Id einer Ladestation.
	///</summary>
	///<param name='chargingStationId'>Id der Ladestation.</param>
	///<returns>Haltestellen-Id der Ladestation</returns>
	CU_HSTDEV StopId getStopIdOfChargingStation(ChargingStationId chargingStationId) const;

private:
	CuStops() : _numOfStops(0), _numOfDepots(0), _numOfChargingStations(0), _stops(0), _depotIds(0), _chargingStationIds(0), _devicePtr(0) {}
	ushort _numOfStops;					// Anzahl der Haltestellen.
	ushort _numOfDepots;				// Anzahl der Depots.
	ushort _numOfChargingStations;		// Anzahl der Ladestationen.
	CuVector1<CuStop> *_stops;
	CuVector1<StopId> *_depotIds;
	CuVector1<StopId> *_chargingStationIds;
	CuStops *_devicePtr;
};
