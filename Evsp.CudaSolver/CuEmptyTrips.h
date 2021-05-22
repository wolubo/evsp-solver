#pragma once

//#include "cuda_runtime.h"

#include "CuVector1.hpp"
#include "EvspLimits.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"


struct /*__align__(16)*/ CuEmptyTrip {
	CU_HSTDEV CuEmptyTrip() {}
	CU_HSTDEV CuEmptyTrip(StopId theFromStopId, StopId theToStopId, DistanceInMeters theDistance, DurationInSeconds theDuration) 
		: fromStopId(theFromStopId), toStopId(theToStopId), distance(theDistance), duration(theDuration) {}
	StopId fromStopId;
	StopId toStopId;
	DistanceInMeters distance;
	DurationInSeconds duration;
};

///<summary>
/// Container für alle Verbindungsfahrten. Optimiert für die Verwendung in Cuda-Kerneln.
///</summary>
class CuEmptyTrips
{
public:
	CuEmptyTrips(int maxNumOfEmptyTrips);
	~CuEmptyTrips();

	CuEmptyTrips* getDevPtr();

	///<summary>
	/// 
	///</summary>
	///<param name="from">Id der Starthaltestelle</param>
	///<param name="to">Id der Zielhaltestelle</param>
	///<param name="dist">Distanz in Metern</param>
	///<param name="rt">Fahrzeit in Sekunden</param>
	void add(StopId from, StopId to, DistanceInMeters dist, DurationInSeconds rt);

	///<summary>
	/// Liefert die Anzahl der Verbindungsfahrten, die im Container gespeichert sind.
	///</summary>
	///<param name=''></param>
	///<returns>Maximale Anzahl der Verbindungsfahrten.</returns>
	CU_HSTDEV int getNumOfEmptyTrips() const;

	///<summary>
	/// Liefert die Id der letzten Verbindungsfahrt, die im Container gespeichert ist.
	///</summary>
	///<param name=''></param>
	///<returns>Maximale Anzahl der Verbindungsfahrten.</returns>
	CU_HSTDEV EmptyTripId lastId() const
	{
		if (_numOfEmptyTrips > 0)
			return EmptyTripId(_numOfEmptyTrips - 1);
		else
			return EmptyTripId::invalid();
	}

	///<summary>
	/// 
	///</summary>
	///<param name='id'>Id der Verbindungsfahrt.</param>
	///<returns></returns>
	CU_HSTDEV const CuEmptyTrip& getEmptyTrip(EmptyTripId id) const;

private:
	CuEmptyTrips() : _numOfEmptyTrips(0), _emptyTrips(), _devicePtr() {}
	ushort _numOfEmptyTrips;
	CuVector1<CuEmptyTrip> *_emptyTrips;
	CuEmptyTrips *_devicePtr;
};

