#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/PointInTime.hpp"
#include "EVSP.BaseClasses/PeriodOfTime.hpp"
#include "EVSP.BaseClasses/Typedefs.h"
#include "EvspLimits.h"
#include "CuStops.h"
#include "CuVehicleTypes.h"
#include "CuEmptyTrips.h"
#include "CuServiceTrips.h"

enum class CuActionType : unsigned char {

	/// <summary>
	/// Der Typ der Aktion ist undefiniert!
	/// </summary>
	INVALID_ACTION,

	/// <summary>
	/// Zwischenzeitliches Aufsuchen des Depots. Ggf. verbunden mit dem Aufladen des Akkus.
	/// </summary>
	VISIT_DEPOT,

	/// <summary>
	/// Durchführen einer Servicefahrt.
	/// </summary>
	SERVICE_TRIP,

	/// <summary>
	/// Durchführen einer Verbindungsfahrt.
	/// </summary>
	EMPTY_TRIP,

	/// <summary>
	/// Aufladen an einer Ladestation.
	/// </summary>
	CHARGE
};


/// <summary>
/// 
///
/// </summary>
class CuAction {
public:

	/// <summary>
	/// Erzeugt eine ungültige Aktion (Typ INVALID_ACTION). Dieser Constructor dient lediglich der Initialisierung innerhalb von
	/// Containern. Er erzeugt unbrauchbare Objekte!
	/// </summary>
	 CuAction();

	 CuAction(const CuAction &other);

	/// <summary>
	/// Erzeugt eine gültige Aktion des Typs VISIT_DEPOT.
	/// </summary>
	/// <param name="stopId">Haltestellen-Id des Depots. Der Aufrufer muss sicherstellen, dass es sich tatsächlich um ein Depot handelt.</param>
	 CuAction(StopId stopId);

	/// <summary>
	/// Erzeugt eine gültige Aktion des Typs CHARGING.
	/// </summary>
	/// <param name="stopId">Haltestellen-Id der Ladestation.</param>
	/// <param name="vehType">Fahrzeugtyp, mit dem die Aktion durchgeführt wird. Dient 
	/// lediglich zur Bestimmung der Werte der Member. Wird nicht im Objekt gespeichert.</param>
	 CuAction(StopId stopId, const CuVehicleType &vehType);

	/// <summary>
	/// Erzeugt eine gültige Aktion des Typs EMPTY_TRIP.
	/// </summary>
	/// <param name="emptyTrip">Daten einer Verbindungsfahrt.</param>
	/// <param name="problem">Dient lediglich dazu, die Werte der Member des Objekts zu bestimmen. Wird nicht im Objekt abgelegt.</param>
	 CuAction(const CuEmptyTrip &emptyTrip);

	/// <summary>
	/// Erzeugt eine gültige Aktion des Typs CuActionType::SERVICE_TRIP.
	/// </summary>
	/// <param name="serviceTripId">Id einer Servicefahrt.</param>
	/// <param name="serviceTrip">Daten einer Servicefahrt.</param>
	/// lediglich zur Bestimmung der Werte der Member. Wird nicht im Objekt gespeichert.</param>
	/// <param name="problem">Dient lediglich dazu, die Werte der Member des Objekts zu bestimmen. Wird nicht im Objekt abgelegt.</param>
	 CuAction(ServiceTripId serviceTripId, const CuServiceTrip &serviceTrip);

	 ~CuAction() {}

	CuAction& operator=(const CuAction &rhs);

	bool operator==(const CuAction &rhs);
	bool operator!=(const CuAction &rhs);

	/// <summary>
	/// Liefert den Typen der Aktion.
	/// </summary>
	/// <returns>Typ der Aktion</returns>
	 CuActionType getType() const { return _type; }

	/// <summary>
	/// Liefert die Id der mit der Aktion assoziierten Servicefahrt.
	/// </summary>
	/// <returns>Id der asssoziierten Servicefahrt oder eine ungültige Id, falls es sich nicht um eine Aktion des Typs CuActionType::SERVICE_TRIP handelt.</returns>
	 ServiceTripId getServiceTripId() const;

	/// <summary>
	/// Liefert die Haltestellen-Id der mit der Aktion assoziierten Ladestation. 
	/// </summary>
	/// <returns>Id der asssoziierten Ladestation oder eine ungültige Id, falls es sich nicht um eine Aktion des Typs CHARGE handelt.</returns>
	 StopId getChargingStationId() const;

	/// <summary>
	/// Liefert die Haltestellen-Id des mit der Aktion assoziierten Depots. 
	/// </summary>
	/// <returns>Id des asssoziierten Depots oder eine ungültige Id, falls es sich nicht um eine Aktion des Typs VISIT_DEPOT handelt.</returns>
	 StopId getDepotId() const;

	/// <summary>
	/// Liefert die Haltestellen-Id der mit der Aktion assoziierten Haltestelle. 
	/// </summary>
	/// <returns>Id der asssoziierten Haltestelle oder eine ungültige Id, falls es sich nicht um eine Aktion des passenden Typs handelt.</returns>
	 StopId getStopId() const;

	 StopId getFromStopId() const { return _fromStopId; }
	 StopId getToStopId() const { return _toStopId; }

	 DistanceInMeters getDistance() const { return _distance; }

	 DurationInSeconds getDuration() const { return _period.getDuration(); }

	/// <summary>
	/// Liefert den Zeitraum, während dessen die Aktion stattfindet.
	/// Enthält nur bei Servicefahrten einen sinnvollen Zeitraum. Dient bei allen anderen Typen
	/// dazu, die Zeitdauer zu speichern (als Zeitraum ab dem Zeitpunkt 0).
	/// </summary>
	 PeriodOfTime getPeriod() const { return _period; }

	 void dump() const;
	 void dumpTime() const;

private:
	/// <summary>
	/// Enthält den Typen der Aktion.
	/// </summary>
	CuActionType _type;

	/// <summary>
	/// Wenn die Aktion vom Typen CuActionType::SERVICE_TRIP wird hier die Id dieser Servicefahrt gespeichert.
	/// </summary>
	ServiceTripId _serviceTripId;

	StopId _fromStopId;
	StopId _toStopId;

	/// <summary>
	/// Zeitraum, während dessen die Aktion stattfindet.
	/// Enthält nur bei Servicefahrten einen sinnvollen Zeitraum. Dient bei allen anderen Typen
	/// dazu, die Zeitdauer zu speichern (als Zeitraum ab dem Zeitpunkt 0).
	/// </summary>
	PeriodOfTime _period;

	DistanceInMeters _distance;
};