#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>

#include "EVSP.BaseClasses/Typedefs.h"


class ServiceTrip;

/// <summary>
/// Repräsentiert eine Buslinie.
/// </summary>
class BusRoute
{
	friend class Problem;
public:
	BusRoute() = delete;
	~BusRoute() {}

	/// <summary>
	/// Erzeugt eine neue leere Buslinie. Also eine Buslinie, die keine Servicefahrten enthält.
	/// </summary>
	/// <param name="code">Liniennummer</param>
	/// <param name="name">Linienbezeichnung</param>
	BusRoute(const std::string &legacyId, const std::string &code, const std::string &name);

	/**
	* Liniennummer
	*/
	std::string getCode();

	/**
	* Linienbezeichnung
	*/
	std::string getName();

	/**
	* Liefert die Anzahl der Servicefahrten, die in dieser Linie enthalten sind.
	*/
	int getNumberOfServiceTrips();

	/**
	* Liefert eine Servicefahrt der Linie.
	* @param index
	*/
	std::shared_ptr<ServiceTrip> getServiceTrip(ServiceTripId id);

	/**
	* Fügt eine Servicefahrt hinzu.
	* @param newServiceTrip
	*/
	void addServiceTrip(std::shared_ptr<ServiceTrip> newServiceTrip);


	std::string toString();

	RouteId getId() const { return _id; }

	void write2file(std::ofstream &txtfile);
	std::string getLegacyId() { return _legacyId; }


#pragma region private_member_variables

private:

	RouteId _id;

	/// <summary>
	/// Liniennummer
	/// </summary>
	std::string _code;

	/// <summary>
	/// Linienbezeichnung
	/// </summary>
	std::string _name;

	/// <summary>
	/// Alle Servicefahrten, die auf der Linie stattfinden.
	/// </summary>
	std::vector<std::shared_ptr<ServiceTrip>> _serviceTrips;

	std::string _legacyId;

#pragma endregion
};
