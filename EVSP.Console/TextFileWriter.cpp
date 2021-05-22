#include <iostream>
#include <fstream>
#include <algorithm>

#include "TextFileWriter.h"
#include "EVSP.BaseClasses/Typedefs.h"


using namespace std;
//using namespace boost;

TextFileWriter::TextFileWriter()
{
}


TextFileWriter::~TextFileWriter()
{
}

#ifdef _DEBUG
void TextFileWriter::writeProblemDefinition(const std::shared_ptr<Problem> &problem, const std::string& filename)
{
	ofstream txtfile;
	txtfile.open(filename);

	writeHeader_STOPPOINT(txtfile);
	const vector<shared_ptr<Stop>>& stops = problem->getStops();
	for (int i = 0; i < stops.size(); i++) {
		shared_ptr<Stop> bs = stops[i];
		bs->write2file(txtfile);
	}

	writeHeader_LINE(txtfile);
	vector<shared_ptr<BusRoute>> busRoutes = problem->getRoutes();
	for (int i = 0; i < busRoutes.size(); i++) {
		shared_ptr<BusRoute> br = busRoutes[i];
		br->write2file(txtfile);
	}

	writeHeader_VEHICLETYPE(txtfile);
	vector<shared_ptr<VehicleType>> vehicleTypes = problem->getVehicleTypes();
	for (int i = 0; i < vehicleTypes.size(); i++) {
		shared_ptr<VehicleType> vt = vehicleTypes[i];
		vt->write2file(txtfile);
	}

	writeHeader_VEHICLETYPEGROUP(txtfile);
	vector<shared_ptr<VehicleTypeGroup>> vehicleTypeGroups = problem->getVehicleTypeGroups();
	vector<string> vehType2vehTypeGrp;
	for (int i = 0; i < vehicleTypeGroups.size(); i++) {
		shared_ptr<VehicleTypeGroup> vtg = vehicleTypeGroups[i];
		vtg->write2file(txtfile);
		//vtg->vehType2vehTypeGrp(vehType2vehTypeGrp, vehicleTypes);
	}

	sort(vehType2vehTypeGrp.begin(), vehType2vehTypeGrp.end());

	writeHeader_VEHTYPETOVEHTYPEGROUP(txtfile);
	for (int i = 0; i < vehType2vehTypeGrp.size(); i++) {
		txtfile << vehType2vehTypeGrp[i] << endl;
	}

	writeHeader_VEHTYPECAPTOSTOPPOINT(txtfile);

	writeHeader_VEHTYPETOCHARGINGSTATION(txtfile);

	writeHeader_SERVICEJOURNEY(txtfile);
	vector<shared_ptr<ServiceTrip>> serviceTrips = problem->getServiceTrips();
	for (int i = 0; i < serviceTrips.size(); i++) {
		shared_ptr<ServiceTrip> st = serviceTrips[i];
		st->write2file(txtfile);
	}

	writeHeader_DEADRUNTIME(txtfile);
	vector<shared_ptr<EmptyTrip>> emptyTrips = problem->getEmptyTrips();
	for (int i = 0; i < emptyTrips.size(); i++) {
		shared_ptr<EmptyTrip> et = emptyTrips[i];
		et->write2file(txtfile);
	}
		
	txtfile.close();
}
#endif

void TextFileWriter::writeHeader_CHARGINGSYSTEM(ofstream &txtfile)
{
	txtfile << "$CHARGINGSYSTEM:ID;Name" << endl;
}


void TextFileWriter::writeHeader_STOPPOINT(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$STOPPOINT:ID;Code;Name;VehCapacityForCharging" << endl;
}


void TextFileWriter::writeHeader_LINE(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$LINE:ID;Code;Name;;;;;;;;" << endl;
}


void TextFileWriter::writeHeader_VEHICLETYPE(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$VEHICLETYPE:ID;Code;Name;VehCharacteristic;VehClass;CurbWeightKg;VehCost;KmCost;HourCost;Capacity;KilowattHour;ConsumptionPerServiceJourneyKm;ConsumptionPerDeadheadKm;RechargingCost;SlowRechargingTime;FastRechargingTime;ChargingSystem" << endl;
}


void TextFileWriter::writeHeader_VEHICLETYPEGROUP(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$VEHICLETYPEGROUP:ID;Code;Name" << endl;
}


void TextFileWriter::writeHeader_VEHTYPETOVEHTYPEGROUP(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$VEHTYPETOVEHTYPEGROUP:VehTypeID;VehTypeGroupID" << endl;
}


void TextFileWriter::writeHeader_VEHTYPECAPTOSTOPPOINT(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$VEHTYPECAPTOSTOPPOINT:VehTypeID;StoppointID;Min;Max;;;;;;;" << endl;
}


void TextFileWriter::writeHeader_VEHTYPETOCHARGINGSTATION(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$VEHTYPETOCHARGINGSTATION:VehTypeID;StoppointID" << endl;
}


void TextFileWriter::writeHeader_SERVICEJOURNEY(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$SERVICEJOURNEY:ID;LineID;FromStopID;ToStopID;DepTime;ArrTime;MinAheadTime;MinLayoverTime;VehTypeGroupID;MaxShiftBackwardSeconds;MaxShiftForwardSeconds;Distance" << endl;
}


void TextFileWriter::writeHeader_DEADRUNTIME(ofstream &txtfile)
{
	txtfile << "*" << endl;
	txtfile << "$DEADRUNTIME:FromStopID;ToStopID;FromTime;ToTime;Distance;RunTime;;;;;" << endl;
}
