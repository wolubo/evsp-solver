#include "VehicleType.h"

#include <iostream>
#include <fstream>

#include "EVSP.BaseClasses/Helper.h"

using namespace std;
//using namespace boost;


VehicleType::VehicleType(const std::string& legacyId, const std::string &theCode, const std::string &theName,
	AmountOfMoney theVehCost, AmountOfMoney theKmCost, AmountOfMoney theHourCost,
	KilowattHour theBatteryCapacity, KilowattHour theConsumptionPerServiceJourneyKm, KilowattHour theConsumptionPerDeadheadKm,
	AmountOfMoney theRechargingCost, DurationInSeconds theRechargingTime,
	//				std::vector<std::shared_ptr<ChargingSystem>> supportedChargingSystems,
	std::string vehCharacteristic, std::string vehClass, std::string curbWeightKg, std::string capacity, std::string slowRechargingTime, std::string fastRechargingTime)
	: _legacyId(legacyId), _code(theCode), _name(theName), _vehCost(theVehCost), _kmCost(theKmCost), _hourCost(theHourCost), _batteryCapacity(theBatteryCapacity), _consumptionPerServiceJourneyKm(theConsumptionPerServiceJourneyKm), _consumptionPerDeadheadKm(theConsumptionPerDeadheadKm), _rechargingCost(theRechargingCost), _rechargingTime(theRechargingTime),
	/*_supportedChargingSystems(supportedChargingSystems),*/ _vehCharacteristic(vehCharacteristic), _vehClass(vehClass), _curbWeightKg(curbWeightKg), _capacity(capacity), _slowRechargingTime(slowRechargingTime), _fastRechargingTime(fastRechargingTime)
{
	if (theCode.length() == 0) throw  invalid_argument("theCode");
	if (theName.length() == 0) throw  invalid_argument("theName");
	if (!theVehCost.isValid()) throw  invalid_argument("theVehCost");
	if (!theKmCost.isValid()) throw  invalid_argument("theKmCost");
	if (!theHourCost.isValid()) throw  invalid_argument("theHourCost");
	if (theBatteryCapacity < 0.0f) throw  invalid_argument("theBatteryCapacity");
	if (theConsumptionPerServiceJourneyKm < 0.0f) throw  invalid_argument("theConsumptionPerServiceJourneyKm");
	if (theConsumptionPerDeadheadKm < 0.0f) throw  invalid_argument("theConsumptionPerDeadheadKm");
	if (!theRechargingCost.isValid()) throw  invalid_argument("theRechargingCost");
	if (!theRechargingTime.isValid()) throw  invalid_argument("theRechargingTime");
}


string VehicleType::getCode() const
{
	return _code;
}


string VehicleType::getName() const
{
	return _name;
}


AmountOfMoney VehicleType::getVehCost() const
{
	return _vehCost;
}


AmountOfMoney VehicleType::getKmCost() const
{
	return _kmCost;
}


AmountOfMoney VehicleType::getHourCost() const
{
	return _hourCost;
}


KilowattHour VehicleType::getBatteryCapacity() const
{
	return _batteryCapacity;
}


KilowattHour VehicleType::getConsumptionPerServiceJourneyKm() const
{
	return _consumptionPerServiceJourneyKm;
}


KilowattHour VehicleType::getConsumptionPerDeadheadKm() const
{
	return _consumptionPerDeadheadKm;
}


AmountOfMoney VehicleType::getRechargingCost() const
{
	return _rechargingCost;
}


DurationInSeconds VehicleType::getRechargingTime() const
{
	return _rechargingTime;
}


string VehicleType::toString()
{
	return "Fahrzeugtyp: Code=" + _code + ", Name=" + _name;
}


#ifdef _DEBUG
void VehicleType::write2file(std::ofstream &txtfile)
{
	//$VEHICLETYPE:ID; Code; Name; VehCharacteristic; VehClass; CurbWeightKg; VehCost; KmCost; HourCost; Capacity; KilowattHour; ConsumptionPerServiceJourneyKm; ConsumptionPerDeadheadKm; RechargingCost; SlowRechargingTime; FastRechargingTime; ChargingSystem

	//string cs = "";
	//for (unsigned int i = 0; i < _supportedChargingSystems.size(); i++)
	//{
	//	if (cs.length() > 0) cs += ",";
	//	cs += _supportedChargingSystems.at(i)->getLegacyId();
	//}

	txtfile
		<< _legacyId << ";"
		<< _code << ";"
		<< _name << ";"
		<< _vehCharacteristic << ";"
		<< _vehClass << ";"
		<< _curbWeightKg << ";"
		<< (int)_vehCost << ";"
		<< (int)_kmCost << ";"
		<< (int)_hourCost << ";"
		<< _capacity << ";"
		<< (float)_batteryCapacity << ";"
		<< (float)_consumptionPerServiceJourneyKm << ";"
		<< (float)_consumptionPerDeadheadKm << ";"
		<< (int)_rechargingCost << ";"
		<< _slowRechargingTime << ";"
		<< _fastRechargingTime << ";"
		<< /*cs <<*/ endl;
}
#endif

//     Code;           Name;     VehCharacteristic; VehClass; CurbWeightKg; VehCost; KmCost; HourCost; Capacity; KilowattHour; ConsumptionPerServiceJourneyKm; ConsumptionPerDeadheadKm; RechargingCost; SlowRechargingTime; FastRechargingTime; ChargingSystem
//- 1; RB_BEV_BYD_12m; BYD_ebus; 1;                 12m;      13300;        100000;  1;      100;      1000;     324;                                1, 296;                         1, 296;                   1000; 18000; 5760; 0, 2
//+ 1; RB_BEV_BYD_12m; BYD_ebus; 1;                 12m;      13300;        100000;  1;      100;      1000;     324;                                1;                              1;                        1000; 18000; 5760; 0, 2

