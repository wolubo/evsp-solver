#include "Stop.h"

#include <iostream>
#include <fstream>
#include <assert.h>

//#include "EVSP.BaseClasses/EvspLimits.h"

using namespace std;
//using namespace boost;


Stop::Stop(const string &legacyId, const string &code, const string &name, bool chargingStation, std::string vehCapacityForCharging)
	: _legacyId(legacyId), _code(code), _name(name), _isChargingStation(chargingStation), _isDepot(false), _vehCapacityForCharging(vehCapacityForCharging)
{
	if (code.length() == 0) throw invalid_argument("code");
	if (name.length() == 0) throw invalid_argument("name");
}


string Stop::getCode() const
{
	return _code;
}


string Stop::getName() const
{
	return _name;
}


bool Stop::isChargingStation() const
{
	return _isChargingStation;
}


bool Stop::isBusStop() const
{
	return !_isDepot;
}


bool Stop::isDepot() const
{
	return _isDepot;
}


void Stop::setDepot(bool isDepot)
{
	_isDepot = isDepot;
}


string Stop::toString()
{
	string retVal = "";
	if (_isDepot)
		retVal += "Depot: ";
	else
		retVal += "Busstop: ";

	retVal += "Code = " + _code + ", Name=" + _name;
	if (_isChargingStation) retVal += " (charging station)";
	return retVal;
}


void Stop::write2file(ofstream &txtfile)
{
	//$STOPPOINT:ID; Code; Name; VehCapacityForCharging
	txtfile << _legacyId << ";" << _code << ";" << _name << ";";
	if (_isChargingStation) {
		txtfile << _vehCapacityForCharging;
	}
	txtfile << endl;
}

