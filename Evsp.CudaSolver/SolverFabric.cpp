#include "SolverFabric.h"

#include <memory>
#include "CuSolutionGeneratorAcoGpu.h"
#include "CuSolutionGeneratorAcoCpu.h"
#include "CuSolutionEvaluatorAco.h"
#include "CuSolutionSelectorAco.h"
#include "CuSolutionPreparatorAco.h"
#include "RandomCpu.h"
#include "RandomGpu.h"
#include "CuAcoSolver.h"
#include "CuSaSolver.h"

SolverFabric::SolverFabric(shared_ptr<Configuration> config, shared_ptr<Problem> problem)
	: _config(config)
{
	_problem = convertProblem(problem, _config->getPlattform());
}


SolverFabric::~SolverFabric()
{
}


std::shared_ptr<CuSolver> SolverFabric::createSaSolver()
{
	shared_ptr<ResultLogger> resultLogger = make_shared<ResultLogger>();

	PlattformConfig plattform = _config->getPlattform();
	int numOfThreads = _config->getNumOfThreads();
	SaParams saParams = _config->getSaParams();

	shared_ptr<CuSaSolver> solver = make_shared<CuSaSolver>(_problem, plattform, numOfThreads, saParams, resultLogger);

	return solver;
}


std::shared_ptr<CuSolver> SolverFabric::createAcoSolver()
{
	shared_ptr<ResultLogger> resultLogger = make_shared<ResultLogger>();

	AcoParams acoParams = _config->getAcoParams();

	AcoQualifiers qualifiers = _config->getQualifiers();
	PlattformConfig plattform = _config->getPlattform();
	bool verbose = _config->isVerboseModeActive();
	int numOfThreads = _config->getNumOfThreads();

	// Aufbau des Konstruktionsgraphen
	shared_ptr<CuConstructionGraph> ptn = make_shared<CuConstructionGraph>(_problem, qualifiers.initialWeight * acoParams.populationSize, plattform, acoParams.performChecks); 

	if (acoParams.printStats) {
		ptn->printStatistic();
	}

	if (acoParams.dumpDecisionNet) {
		ptn->dumpDecisionNet(_problem);
	}

	std::shared_ptr<CuAcoSolver> solver = make_shared<CuAcoSolver>(_problem, ptn, acoParams, qualifiers, plattform, numOfThreads, verbose, resultLogger);

	return solver;
}


std::shared_ptr<CuProblem> SolverFabric::convertProblem(shared_ptr<Problem> problem, PlattformConfig plattform) {

	CuStops *stops = 0;
	{
		// Haltestellen konvertieren
		const vector<shared_ptr<Stop>> model = problem->getStops();
		stops = new CuStops((int)model.size(), (int)problem->getDepots().size(), (int)problem->getChargingStations().size());
		vector<shared_ptr<Stop>>::const_iterator iter = model.begin();
		while (iter != model.end()) {
			shared_ptr<Stop> c = (*iter);
			bool isDepot = c->isDepot();
			bool isChargingStation = c->isChargingStation();
			stops->add(isDepot, isChargingStation);
			iter++;
		}
		assert(model.size() == stops->getNumOfStops());
	}

	CuEmptyTrips *emptyTrips = 0;
	{
		// Verbindungsfahrten konvertieren
		const vector<shared_ptr<EmptyTrip>> model = problem->getEmptyTrips();
		emptyTrips = new CuEmptyTrips((int)model.size());
		vector<shared_ptr<EmptyTrip>>::const_iterator iter = model.begin();
		while (iter != model.end()) {
			shared_ptr<EmptyTrip> c = (*iter);
			StopId fromStop = c->getFromStop()->getId();
			StopId toStop = c->getToStop()->getId();
			DistanceInMeters distance = c->getDistance();
			DurationInSeconds duration = c->getDuration();
			emptyTrips->add(fromStop, toStop, distance, duration);
			iter++;
		}
		assert(model.size() == emptyTrips->getNumOfEmptyTrips());
	}

	CuServiceTrips *serviceTrips = 0;
	{
		// Servicefahrten konvertieren
		const vector<shared_ptr<ServiceTrip>> model = problem->getServiceTrips();
		serviceTrips = new CuServiceTrips((int)model.size());
		vector<shared_ptr<ServiceTrip>>::const_iterator iter = model.begin();
		while (iter != model.end()) {
			shared_ptr<ServiceTrip> c = (*iter);
			StopId fromStop = c->getFromStop()->getId();
			StopId toStop = c->getToStop()->getId();
			DistanceInMeters distance = c->getDistance();
			VehicleTypeGroupId vtg = c->getVehicleTypeGroup()->getId();
			PointInTime departure = c->getScheduledTime().getBegin();
			PointInTime arrival = c->getScheduledTime().getEnd();
			serviceTrips->add(fromStop, toStop, distance, vtg, departure, arrival);
			iter++;
		}
		assert(model.size() == serviceTrips->getNumOfServiceTrips());
	}

	CuVehicleTypes *vehicleTypes = 0;
	{
		// Fahrzeugtypen konvertieren
		const vector<shared_ptr<VehicleType>> model = problem->getVehicleTypes();
		vehicleTypes = new CuVehicleTypes((int)model.size());
		vector<shared_ptr<VehicleType>>::const_iterator iter = model.begin();
		while (iter != model.end()) {
			shared_ptr<VehicleType> c = (*iter);
			AmountOfMoney vehCost = c->getVehCost();
			AmountOfMoney kmCost = c->getKmCost();
			AmountOfMoney hourCost = c->getHourCost();
			KilowattHour battCapacity = c->getBatteryCapacity();
			KilowattHour consServKm = c->getConsumptionPerServiceJourneyKm();
			KilowattHour consDeadKm = c->getConsumptionPerDeadheadKm();
			AmountOfMoney rechargingCost = c->getRechargingCost();
			DurationInSeconds rechargingTime = c->getRechargingTime();
			vehicleTypes->add(vehCost, kmCost, hourCost, battCapacity, consServKm, consDeadKm, rechargingCost, rechargingTime);
			iter++;
		}
		assert(model.size() == vehicleTypes->getNumOfVehicleTypes());
	}

	CuVehicleTypeGroups *vehicleTypeGroups = 0;
	{
		// Fahrzeugtypgruppen konvertieren
		const vector<shared_ptr<VehicleTypeGroup>> model = problem->getVehicleTypeGroups();
		vehicleTypeGroups = new CuVehicleTypeGroups((int)model.size(), vehicleTypes->getNumOfVehicleTypes());
		vector<shared_ptr<VehicleTypeGroup>>::const_iterator vtgIter = model.begin();
		while (vtgIter != model.end()) {
			VehicleTypeGroupId vtg_id = VehicleTypeGroupId(vehicleTypeGroups->getNumOfVehicleTypeGroups());
			vehicleTypeGroups->addGroup();
			shared_ptr<VehicleTypeGroup> c = (*vtgIter);
			VehicleTypeId lastId = VehicleTypeId(c->getNumberOfVehicleTypes());
			for (VehicleTypeId i(0); i < lastId; i++) {
				shared_ptr<VehicleType> vt = c->getVehicleType(i);
				vehicleTypeGroups->addType(vtg_id, vt->getId());
			}
			vtgIter++;
		}

		assert(model.size() == vehicleTypeGroups->getNumOfVehicleTypeGroups());
	}

	return make_shared<CuProblem>(stops, emptyTrips, serviceTrips, vehicleTypes, vehicleTypeGroups, plattform, _config->getPerformChecks());
}