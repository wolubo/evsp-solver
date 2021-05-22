#pragma warning(disable: 4267)

#include "Problem.h"

#include <assert.h>
#include <iostream>
#include <iomanip>
//#include <limits>

#include "EVSP.BaseClasses/Typedefs.h"

using namespace std;



Problem::Problem()
	: _emptyTrips(), _vehicleTypes(), _vehicleTypeGroups(), _routes(), _serviceTrips(), _stops(),
	_avgEmptyTripDistance(0), _avgEmptyTripDuration(0), _avgServiceTripDistance(0), _avgServiceTripDuration(0)
{

}


Problem::~Problem()
{
}


void Problem::addRoute(shared_ptr<BusRoute> newRoute) {
	assert(newRoute);
	newRoute->_id = RouteId(_routes.size());
	_routes.push_back(newRoute);
}


void Problem::addServiceTrip(shared_ptr<ServiceTrip> newServiceTrip) {
	assert(newServiceTrip);
	newServiceTrip->_id = ServiceTripId(_serviceTrips.size());
	_serviceTrips.push_back(newServiceTrip);
	_avgServiceTripDistance = DistanceInMeters::invalid();
	_avgServiceTripDuration = DurationInSeconds::invalid();
}


void Problem::addStop(shared_ptr<Stop> newStop) {
	assert(newStop);
	newStop->_id = StopId(_stops.size());
	_stops.push_back(newStop);
}


void Problem::addEmptyTrip(shared_ptr<EmptyTrip> newEmptyTrip) {
	assert(newEmptyTrip);
	newEmptyTrip->_id = EmptyTripId(_emptyTrips.size());
	_emptyTrips.push_back(newEmptyTrip);
	_avgEmptyTripDistance = DistanceInMeters::invalid();
	_avgEmptyTripDuration = DurationInSeconds::invalid();
}


void Problem::addVehicleType(shared_ptr<VehicleType> newVehicleType) {
	assert(newVehicleType);
	newVehicleType->_id = VehicleTypeId(_vehicleTypes.size());
	_vehicleTypes.push_back(newVehicleType);
}


void Problem::addVehicleTypeGroup(shared_ptr<VehicleTypeGroup> newVehicleTypeGroup) {
	assert(newVehicleTypeGroup);
	newVehicleTypeGroup->_id = VehicleTypeGroupId(_vehicleTypeGroups.size());
	_vehicleTypeGroups.push_back(newVehicleTypeGroup);
}


const vector<shared_ptr<BusRoute>>& Problem::getRoutes() const
{
	return _routes;
}


const vector<shared_ptr<ServiceTrip>>& Problem::getServiceTrips() const
{
	return _serviceTrips;
}


const vector<shared_ptr<Stop>>& Problem::getStops() const
{
	return _stops;
}


vector<shared_ptr<Stop>> Problem::getBusStops() const
{
	vector<shared_ptr<Stop>> retVal = vector<shared_ptr<Stop>>();
	vector<shared_ptr<Stop>>::const_iterator iter = _stops.begin();
	while (iter != _stops.end()) {
		if ((*iter)->isBusStop()) {
			retVal.push_back((*iter));
		}
		iter++;
	}
	return retVal;
}


vector<shared_ptr<Stop>> Problem::getDepots() const
{
	vector<shared_ptr<Stop>> retVal = vector<shared_ptr<Stop>>();
	vector<shared_ptr<Stop>>::const_iterator iter = _stops.begin();
	while (iter != _stops.end()) {
		if ((*iter)->isDepot()) {
			retVal.push_back((*iter));
		}
		iter++;
	}
	return retVal;
}


vector<shared_ptr<Stop>> Problem::getChargingStations() const
{
	vector<shared_ptr<Stop>> retVal = vector<shared_ptr<Stop>>();
	vector<shared_ptr<Stop>>::const_iterator iter = _stops.begin();
	while (iter != _stops.end()) {
		if ((*iter)->isChargingStation()) {
			retVal.push_back((*iter));
		}
		iter++;
	}
	return retVal;
}


const vector<shared_ptr<EmptyTrip>>& Problem::getEmptyTrips() const
{
	return _emptyTrips;
}


const vector<shared_ptr<VehicleType>>& Problem::getVehicleTypes() const
{
	return _vehicleTypes;
}


const vector<shared_ptr<VehicleTypeGroup>>& Problem::getVehicleTypeGroups() const
{
	return _vehicleTypeGroups;
}


DistanceInMeters Problem::getAvgEmptyTripDistance()
{
	if (!_avgEmptyTripDistance.isValid()) computeEmptyTripAvg();
	return _avgEmptyTripDistance;
}


DurationInSeconds Problem::getAvgEmptyTripDuration()
{
	if (!_avgEmptyTripDuration.isValid()) computeEmptyTripAvg();
	return _avgEmptyTripDuration;
}


DistanceInMeters Problem::getAvgServiceTripDistance()
{
	if (!_avgServiceTripDistance.isValid()) computeServiceTripAvg();
	return _avgServiceTripDistance;
}


DurationInSeconds Problem::getAvgServiceTripDuration()
{
	if (!_avgServiceTripDuration.isValid()) computeServiceTripAvg();
	return _avgServiceTripDuration;
}


void Problem::computeEmptyTripAvg()
{
	DistanceInMeters sum_distance(0);
	DurationInSeconds sum_runtime(0);

	vector<std::shared_ptr<EmptyTrip>>::const_iterator et_iter = _emptyTrips.begin();
	while (et_iter != _emptyTrips.end()) {
		sum_distance = sum_distance + (*et_iter)->getDistance();
		sum_runtime = sum_runtime + (*et_iter)->getDuration();
		et_iter++;
	}

	_avgEmptyTripDistance = DistanceInMeters((int)sum_distance / _emptyTrips.size());
	_avgEmptyTripDuration = DurationInSeconds((int)sum_runtime / _emptyTrips.size());
}


void Problem::computeServiceTripAvg()
{
	DistanceInMeters sum_distance = DistanceInMeters(0);
	DurationInSeconds sum_runtime = DurationInSeconds(0);

	vector<std::shared_ptr<ServiceTrip>>::const_iterator st_iter = _serviceTrips.begin();
	while (st_iter != _serviceTrips.end()) {
		sum_distance = sum_distance + (*st_iter)->getDistance();
		sum_runtime = sum_runtime + (*st_iter)->getScheduledTime().getDuration();
		st_iter++;
	}

	_avgServiceTripDistance = DistanceInMeters((int)sum_distance / _serviceTrips.size());
	_avgServiceTripDuration = DurationInSeconds((int)sum_runtime / _serviceTrips.size());
}


void Problem::printStatistic()
{
	float avg_st_duration = (int)getAvgServiceTripDuration() / 60.0f;
	float avg_st_distance = (int)getAvgServiceTripDistance() / 1000.0f;
	float avg_et_duration = (int)getAvgEmptyTripDuration() / 60.0f;
	float avg_et_distance = (int)getAvgEmptyTripDistance() / 1000.0f;

	cout << endl;
	cout << "Statistik:" << endl;
	cout << "Haltestellen:       " << getStops().size() << endl;
	cout << "Bushaltestellen:    " << getBusStops().size() << endl;
	cout << "Depots:             " << getDepots().size() << endl;
	cout << "Servicefahrten:     " << _serviceTrips.size() << " (durchschn. Dauer: " << fixed << setprecision(0) << avg_st_duration << " Min., durchschn. Stecke: " << avg_st_distance << "km)" << endl;
	cout << "Verbindungsfahrten: " << _emptyTrips.size() << " (durchschn. Dauer: " << avg_et_duration << " Min., durchschn. Stecke: " << avg_et_distance << "km)" << endl;
	cout << "Buslinien:          " << _routes.size() << endl;
	cout << "Fahrzeugtypen:      " << _vehicleTypes.size() << endl;
	cout << "Fahrzeugtypgruppen: " << _vehicleTypeGroups.size() << endl;
	cout << "Ladestationen:      " << getChargingStations().size() << endl;
	cout << endl;
}


void Problem::checkEmptyTrips()
{
	vector<std::shared_ptr<EmptyTrip>>::const_iterator et_iter = _emptyTrips.begin();
	bool change = false;
	while (et_iter != _emptyTrips.end()) {
		if ((*et_iter)->_distance == DistanceInMeters(0)) {
			(*et_iter)->_distance = getAvgEmptyTripDistance();
			change = true;
		}
		if ((*et_iter)->_duration == DurationInSeconds(0)) {
			(*et_iter)->_duration = getAvgEmptyTripDuration();
			change = true;
		}
		et_iter++;
	}
	if (change) {
		cerr << "Warnung: Die Eingabedatei enthält mindestens eine Verbindungsfahrt mit ungültiger Laufzeit bzw. ungültiger Entfernung. Die ungültigen Angaben wurden durch Durchschnittswerte ersetzt." << endl;
	}
}


void Problem::randomizeChargingStations(int ratio)
{
	if (ratio == 0) {
		cout << "Hinweis: Über die Depots hinaus gibt es keine weiteren Ladestationen." << endl;
	}
	else if (ratio == 100) {
		cout << "Hinweis: Alle Bushaltestellen sind Ladestationen." << endl;
		for (int i = 0; i < _stops.size(); i++) {
			_stops[i]->_isChargingStation = true;
		}
	}
	else if (ratio > 0 && ratio < 100) {
		cout << "Hinweis: Ladestationen werden zufällig festgelegt (" << ratio << "% aller Bushaltestellen)" << endl;
		vector<shared_ptr<Stop>> busStops = getBusStops();
		unsigned int numOfBusStops = busStops.size();
		unsigned int numOfChargingStations = (ratio * numOfBusStops) / 100;
		vector<int> choice;
		for (unsigned int i = 0; i < numOfBusStops; i++) {
			choice.push_back(i);
		}
		for (unsigned int n = 0; n < numOfChargingStations; n++) {
			int r = rand() % numOfBusStops;
			int index = choice[r];
			busStops[index]->_isChargingStation = true;
			choice[r] = choice[numOfBusStops - 1];
			numOfBusStops--;
		}
	}

}


void Problem::solveForCombustionVehicles()
{
	for (int i = 0; i < _stops.size(); i++) {
		_stops[i]->_isChargingStation = false;
	}
	for (int i = 0; i < _vehicleTypes.size(); i++) {
		_vehicleTypes[i]->_consumptionPerDeadheadKm = KilowattHour(0.0f);
		_vehicleTypes[i]->_consumptionPerServiceJourneyKm = KilowattHour(0.0f);
	}
	for (int i = 0; i < _vehicleTypes.size(); i++) {
		_vehicleTypes[i]->_batteryCapacity = KilowattHour(std::numeric_limits<float>::max());
	}
}


void Problem::checkServiceTrips(bool verbose)
{
	// Connections-Matrix aufbauen (welche Haltestellen sind über Verbindungsfahrten miteinander verknüpft?).
	const int size = _stops.size();
	EmptyTripId* connections = new EmptyTripId[size*size];
	for (int r = 0; r < size; r++) {
		for (int c = 0; c < size; c++) {
			connections[r*size + c] = EmptyTripId::invalid();
		}
	}

	vector<std::shared_ptr<EmptyTrip>>::const_iterator et_iter = _emptyTrips.begin();
	while (et_iter != _emptyTrips.end()) {
		StopId from = (*et_iter)->getFromStop()->getId();
		StopId to = (*et_iter)->getToStop()->getId();
		connections[(short)from*size + (short)to] = (*et_iter)->getId();
		et_iter++;
	}

	// Depotliste aufbauen
	vector<std::shared_ptr<Stop>> depots;
	vector<std::shared_ptr<Stop>>::const_iterator s_iter = _stops.begin();
	while (s_iter != _stops.end()) {
		if ((*s_iter)->isDepot()) {
			depots.push_back((*s_iter));
		}
		s_iter++;
	}

	unsigned short newEmptyTripCounter = 0;

	// Für jedes Depot und jede Servicefahrt prüfen: Kann jede Servicefahrt von jedem Depot aus starten und auch in 
	// jedem Depot enden? Gibt es also immer eine Verbindungsfahrt?
	vector<std::shared_ptr<ServiceTrip>>::iterator st_iter = _serviceTrips.begin();
	while (st_iter != _serviceTrips.end()) {
		StopId st_from = (*st_iter)->getFromStop()->getId();
		StopId st_to = (*st_iter)->getToStop()->getId();

		s_iter = depots.begin();
		while (s_iter != depots.end()) {
			StopId depotId = (*s_iter)->getId();

			// Gibt es eine Ausrückfahrt von jedem Depot zur Starthaltestelle der Servicefahrt?
			if (!(depotId == st_from || connections[(short)depotId*size + (short)st_from].isValid())) {
				// Ausrückfahrt fehlt! Lege eine Ausrückfahrt mit durchschnittlicher Länge und Laufzeit an.
				shared_ptr<EmptyTrip> newEmptyTrip = make_shared<EmptyTrip>((*s_iter), (*st_iter)->getFromStop(), getAvgEmptyTripDistance(), getAvgEmptyTripDuration(), "", "");
				EmptyTripId newId = EmptyTripId(_emptyTrips.size());
				newEmptyTrip->_id = newId;
				_emptyTrips.push_back(newEmptyTrip);
				assert(_emptyTrips[(short)newId] == newEmptyTrip);
				connections[(short)depotId*size + (short)st_from] = newId;
				newEmptyTripCounter++;
				if (verbose) {
					cout << "Neue Ausrückfahrt: " << newEmptyTrip->toString() << endl;
				}
			}

			// Gibt es eine Einrückfahrt von der Endhaltestelle der Servicefahrt zurück zu jedem Depot?
			if (!(depotId == st_to || connections[(short)st_to*size + (short)depotId].isValid())) {
				// Einrückfahrt fehlt! Lege eine Einrückfahrt mit durchschnittlicher Länge und Laufzeit an.
				shared_ptr<EmptyTrip> newEmptyTrip = make_shared<EmptyTrip>((*st_iter)->getToStop(), (*s_iter), getAvgEmptyTripDistance(), getAvgEmptyTripDuration(), "", "");
				EmptyTripId newId = EmptyTripId(_emptyTrips.size());
				newEmptyTrip->_id = newId;
				_emptyTrips.push_back(newEmptyTrip);
				assert(_emptyTrips[(short)newId] == newEmptyTrip);
				connections[(short)st_to*size + (short)depotId] = newId;
				newEmptyTripCounter++;
				if (verbose) {
					cout << "Neue Einrückfahrt: " << newEmptyTrip->toString() << endl;
				}
			}
			s_iter++;
		}
		st_iter++;
	}

	if (newEmptyTripCounter > 0) {
		cout << "Hinweis: Es wurden" << newEmptyTripCounter << " Verbindungsfahrten hinzugefügt, da nicht alle Servicefahrten an jedes Depot angebunden waren." << endl;
	}
}

