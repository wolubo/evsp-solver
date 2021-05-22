#pragma once

//#include "cuda_runtime.h"

#include "EvspLimits.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/PointInTime.hpp"
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "CuVector1.hpp"



struct /*__align__(16)*/ CuServiceTrip {
	CU_HSTDEV CuServiceTrip() {}
	CU_HSTDEV ~CuServiceTrip() {}
	StopId fromStopId;
	StopId toStopId;
	DistanceInMeters distance;
	VehicleTypeGroupId vehicleTypeGroupId;
	PointInTime departure;
	PointInTime arrival;

	CU_HSTDEV DurationInSeconds getDuration() const { return DurationInSeconds((int)arrival - (int)departure); }
};


class CuServiceTrips
{

public:

	CuServiceTrips(int maxNumOfServiceTrips);
	~CuServiceTrips();

	CuServiceTrips* getDevPtr();

	///<summary>
	/// Fügt eine neue Servicefahrt hinzu.
	///</summary>
	///<param name="fromStopId"></param>
	///<param name="toStopId"></param>
	///<param name="distance"></param>
	///<param name="vehicleTypeGroupId"></param>
	///<param name="departure"></param>
	///<param name="arrival"></param>
	void add(StopId fromStopId, StopId toStopId, DistanceInMeters distance, VehicleTypeGroupId vehicleTypeGroupId, PointInTime departure, PointInTime arrival);

	///<summary>
	/// Liefert die Anzahl der Servicefahrten.
	///</summary>
	///<returns>Aktuelle Anzahl der Servicefahrten.</returns>
	CU_HSTDEV int getNumOfServiceTrips() const;

	///<summary>
	/// 
	///</summary>
	///<param name='serviceTripId'>Id der Servicefahrt.</param>
	///<returns></returns>
	CU_HSTDEV const CuServiceTrip& getServiceTrip(ServiceTripId serviceTripId) const;

	///<summary>
	/// Liefert den Zeitpunkt, zu dem die erste Servicefahrt eines Betriebstags startet. 
	/// Also den frühesten Wert für 'CuServiceTrip.departure', der in der Menge der
	/// Servicefahrten vorkommt.
	///</summary>
	CU_HSTDEV PointInTime getEarliestStartTime() const { return _earliestStartTime; }

	///<summary>
	/// Liefert den Zeitpunkt, zu dem die letzte Servicefahrt eines Betriebstags startet. 
	/// Also den spätesten Wert für 'CuServiceTrip.departure', der in der Menge der
	/// Servicefahrten vorkommt.
	///</summary>
	CU_HSTDEV PointInTime getLatestStartTime() const { return _latestStartTime; }

private:
	CuServiceTrips():_numOfServiceTrips(0), _devicePtr(), _serviceTrips() {}
	ushort _numOfServiceTrips;
	CuVector1<CuServiceTrip> *_serviceTrips;
	CuServiceTrips *_devicePtr;

	// Früheste Abfahrt-Zeit einer Servicefahrt an einem Betriebstag. 
	PointInTime _earliestStartTime;

	// Spästeste Abfahrt-Zeit einer Servicefahrt an einem Betriebstag. 
	PointInTime _latestStartTime;
};
