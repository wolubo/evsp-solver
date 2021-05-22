#include <iostream>
#include <fstream>

#include "VehicleTypeGroup.h"
//#include "EVSP.BaseClasses/EvspLimits.h"

using namespace std;
//using namespace boost;
//using namespace WoBo::EVSP::BaseClasses;





VehicleTypeGroup::VehicleTypeGroup(const string &legacyId, const std::string &theCode, const std::string &theName)
	: _legacyId(legacyId), _code(theCode), _name(theName)
{
	if (theCode.length() == 0) throw  invalid_argument("theCode");
	if (theName.length() == 0) throw  invalid_argument("theName");
}


void VehicleTypeGroup::addVehicleType(std::shared_ptr<VehicleType> vehicleType)
{
	if (vehicleType == 0) throw  range_error("vehicleType");
	_vehicleTypes.push_back(vehicleType);
}


string VehicleTypeGroup::getCode() const
{
	return _code;
}


string VehicleTypeGroup::getName() const
{
	return _name;
}


int VehicleTypeGroup::getNumberOfVehicleTypes() const
{
	return (int)_vehicleTypes.size();
}


std::shared_ptr<VehicleType> VehicleTypeGroup::getVehicleType(VehicleTypeId id) const
{
	return _vehicleTypes.at((short)id);
}


std::string VehicleTypeGroup::toString()
{
	return "VehicleTypeGroup: Code=" + _code + ", Name=" + _name;
}


void VehicleTypeGroup::write2file(std::ofstream &txtfile)
{
	//$VEHICLETYPEGROUP:ID; Code; Name
	txtfile << _legacyId << ";" << _code << ";" << _name << endl;
}

//#ifdef _DEBUG
//			void VehicleTypeGroup::vehType2vehTypeGrp(vector<string> &result, vector<VehicleType> vehTypes)
//			{
//				string s;
//				for (unsigned int i = 0; i < _vehicleTypes.size(); i++)
//				{
//					s = vehTypes[_vehicleTypes[i]].getLegacyId() + ";" + _legacyId;
//					result.push_back(s);
//				}
//			}
//#endif

