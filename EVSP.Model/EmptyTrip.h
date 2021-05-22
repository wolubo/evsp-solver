#pragma once


#include <string>
#include <map>
#include <vector>
#include <memory>

#include "EVSP.BaseClasses/Typedefs.h"
#include "Stop.h"


/// <summary>
/// Repräsentiert eine Verbindungsfahrt.
/// </summary>
class EmptyTrip
{
	friend class Problem;
public:
	EmptyTrip() = delete;
	~EmptyTrip() {}

	/// <summary>
	/// Erzeugt eine neue Verbindungsfahrt.
	/// </summary>
	/// <param name="fromStop">Haltepunkt, an dem die Verbindungsfahrt startet.</param>
	/// <param name="toStop">Haltepunkt, an dem die Verbindungsfahrt endet.</param>
	/// <param name="distance">Distanz zwischen den verbundenen Haltestellen</param>
	/// <param name="duration">Fahrtdauer der Verbindungsfahrt</param>
	EmptyTrip(std::shared_ptr<Stop> fromStop, std::shared_ptr<Stop> toStop,
		DistanceInMeters distance, DurationInSeconds duration,
		std::string fromTime, std::string toTime);

	/// <summary>
	/// Distanz zwischen den verbundenen Haltestellen in Metern.
	/// </summary>
	DistanceInMeters getDistance() const;

	/// <summary>
	/// Fahrtdauer der Verbindungsfahrt in Sekunden.
	/// </summary>
	DurationInSeconds getDuration() const;

	/// <summary>
	/// Haltepunkt, an dem die Verbindungsfahrt startet.
	/// </summary>
	std::shared_ptr<Stop> getFromStop() const;

	/// <summary>
	/// Haltepunkt, an dem die Verbindungsfahrt endet.
	/// </summary>
	std::shared_ptr<Stop> getToStop() const;

	std::string toString();

	EmptyTripId getId() const { return _id; }

#ifdef _DEBUG
	void EmptyTrip::write2file(std::ofstream &txtfile);
#endif

#pragma region private_member_variables
private:

	EmptyTripId _id;

	/// <summary>
	/// Distanz zwischen den verbundenen Haltestellen in Metern.
	/// </summary>
	DistanceInMeters _distance;

	/// <summary>
	/// Fahrtdauer der Verbindungsfahrt in Sekunden.
	/// </summary>
	DurationInSeconds _duration;

	/// <summary>
	/// Haltepunkt, an dem die Verbindungsfahrt startet.
	/// </summary>
	std::shared_ptr<Stop> _fromStop;

	/// <summary>
	/// Haltepunkt, an dem die Verbindungsfahrt endet.
	/// </summary>
	std::shared_ptr<Stop> _toStop;

#pragma endregion

#pragma region unsupported_attributes
	// Die folgenden Attribute sind in den Input-Files enthalten, werden aber vom System derzeit (noch) nicht unterstützt.
	std::string _fromTime;
	std::string _toTime;
#pragma endregion

};
