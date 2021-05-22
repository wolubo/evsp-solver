#pragma once


#include <string>
#include <map>
#include <vector>
#include <memory>

#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/PeriodOfTime.hpp"
#include "EVSP.BaseClasses/PointInTime.hpp"
#include "Stop.h"
#include "VehicleTypeGroup.h"


class BusRoute;

/// <summary>
/// Repräsentiert eine Fahrplanfahrt. Also eine Servicefahrt mit Passagieren auf einer Buslinie von einer Haltestelle zur nächsten Haltestelle.
/// </summary>
class ServiceTrip
{
	friend class Problem;
public:
	ServiceTrip() = delete;
	ServiceTrip(
		const std::string& legacyId,
		std::shared_ptr<BusRoute> route,
		std::shared_ptr<Stop>  fromStop,
		std::shared_ptr<Stop>  toStop,
		PointInTime depTime, PointInTime arrTime,
		std::shared_ptr<VehicleTypeGroup> vehTypeGroup,
		DistanceInMeters distance,
		std::string minAheadTime, std::string minLayoverTime, std::string maxShiftBackwardSeconds, std::string maxShiftForwardSeconds);
	~ServiceTrip() {}

	/// <summary>
	/// Liefert true, wenn die Servicefahrt an der Haltestelle ended, an der sie begonnen hat ("Rundfahrt").
	/// </summary>
	bool isRoundtrip();

	/// <summary>
	/// Zeitraum, während dessen die Servicefahrt stattfindet.
	/// ScheduledTime.Begin: Abfahrtszeit an der Starthaltestelle.
	/// ScheduledTime.End:   Ankunftszeit an der Zielhaltestelle.
	/// </summary>
	PeriodOfTime getScheduledTime() const;

	/// <summary>
	/// Entfernung zwischen der Starthaltestelle und der Zielhaltestelle.
	/// </summary>
	DistanceInMeters getDistance() const;

	/// <summary>
	/// Starthaltestelle: Hier startet die Fahrplanfahrt.
	/// </summary>
	std::shared_ptr<Stop> getFromStop() const;

	/// <summary>
	/// Zielhaltestelle: Hier endet die Fahrplanfahrt.
	/// </summary>
	std::shared_ptr<Stop> getToStop() const;

	/// <summary>
	/// Verweis auf die Linie, deren Bestandteil die Fahrplanfahrt ist.
	/// </summary>
	std::shared_ptr<BusRoute> getRoute() const;

	/// <summary>
	/// Spezifiziert die Fahrzeugtypgruppe, mit der die Fahrplanfahrt bedient werden kann.
	/// </summary>
	std::shared_ptr<VehicleTypeGroup> getVehicleTypeGroup() const;

	std::string toString();

	ServiceTripId getId() const { return _id; }

	void write2file(std::ofstream &txtfile);
	std::string getLegacyId() { return _legacyId; }

#pragma region private_member_variables
private:

	ServiceTripId _id;

	/// <summary>
	/// Zeitraum, während dessen die Servicefahrt stattfindet.
	/// ScheduledTime.Begin: Abfahrtszeit an der Starthaltestelle.
	/// ScheduledTime.End:   Ankunftszeit an der Zielhaltestelle.
	/// </summary>
	PeriodOfTime _scheduledTime;

	/// <summary>
	/// Starthaltestelle: Hier startet die Fahrplanfahrt.
	/// </summary>
	std::shared_ptr<Stop> _fromStop;

	/// <summary>
	/// Zielhaltestelle: Hier endet die Fahrplanfahrt.
	/// </summary>
	std::shared_ptr<Stop> _toStop;

	/// <summary>
	/// Verweis auf die Linie, deren Bestandteil die Fahrplanfahrt ist.
	/// </summary>
	std::shared_ptr<BusRoute> _route;

	/// <summary>
	/// Spezifiziert die Fahrzeugtypgruppe, mit der die Fahrplanfahrt bedient werden kann.
	/// </summary>
	std::shared_ptr<VehicleTypeGroup> _vehicleTypeGroup;

	/// <summary>
	/// Entfernung zwischen der Starthaltestelle und der Zielhaltestelle in Metern.
	/// </summary>
	DistanceInMeters _distance;

	std::string _legacyId;

#pragma endregion

#pragma region unsupported_attributes
	// Die folgenden Attribute sind in den Input-Files enthalten, werden aber vom System derzeit (noch) nicht unterstützt.
	std::string _minAheadTime;
	std::string _minLayoverTime;
	std::string _maxShiftBackwardSeconds;
	std::string _maxShiftForwardSeconds;
#pragma endregion

};

