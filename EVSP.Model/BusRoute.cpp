#include <iostream>
#include <fstream>

#include "BusRoute.h"
#include "ServiceTrip.h"

using namespace std;
//using namespace boost;



BusRoute::BusRoute(const string &legacyId, const string &code, const string &name)
	:_legacyId(legacyId), _code(code), _name(name)
{
	if (code.length() == 0) throw invalid_argument("code");
	if (name.length() == 0) throw invalid_argument("name");
}


string BusRoute::getCode() {
	return _code;
}


string BusRoute::getName() {
	return _name;
}


int BusRoute::getNumberOfServiceTrips() {
	return (int)_serviceTrips.size();
}


std::shared_ptr<ServiceTrip> BusRoute::getServiceTrip(ServiceTripId id)
{
	return _serviceTrips.at((short)id);
}


void BusRoute::addServiceTrip(std::shared_ptr<ServiceTrip> newServiceTrip) {
	if (newServiceTrip == 0) throw  invalid_argument("newServiceTrip is 0!");
	_serviceTrips.push_back(newServiceTrip);
}


std::string BusRoute::toString()
{
	return "Buslinie: Code=" + _code + ", Name=" + _name + " (Enthält " + to_string(_serviceTrips.size()) + " Servicefahrten)";
}


void BusRoute::write2file(std::ofstream &txtfile)
{
	//$LINE:ID; Code; Name;;;;;;;;
	txtfile << _legacyId << ";" << _code << ";" << _name << ";;;;;;;;" << endl;
}

