#include "CuSolutionEvaluatorAco.h"

#include <assert.h>

#include "device_launch_parameters.h"
#include "CuPlans.h"
#include "CuEvaluationResult.h"
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/StopWatch.h"
#include "CuProblem.h"
#include "CuConstructionGraph.h"


CuSolutionEvaluatorAco::State::State() {
	currStop = StopId::invalid();
	totalNetCost = AmountOfMoney(0);
	totalGrossCost = AmountOfMoney(0);
	currentTime = PointInTime::invalid();
	circulationStartTime = PointInTime::invalid();
}


CuSolutionEvaluatorAco::CuSolutionEvaluatorAco(shared_ptr<CuConstructionGraph> ptn, shared_ptr<CuProblem> problem, PlattformConfig plattform, int populationSize, int maxNumOfCirculations)
	: _ptn(ptn), _problem(problem), _plattform(plattform), _results(new CuEvaluationResult(populationSize, maxNumOfCirculations))
{
}


CuSolutionEvaluatorAco::~CuSolutionEvaluatorAco()
{
}


shared_ptr<CuEvaluationResult> CuSolutionEvaluatorAco::runOnCpu(shared_ptr<CuPlans> solutions)
{
	Stopwatch stopwatch;
	stopwatch.start();

	for (int i = 0; i < solutions->getNumOfSolutions(); i++) {
		analyseSolution(i, solutions.get(), _ptn.get(), _problem.get(), _results.get());
	}

	_results->copyToDevice();

	stopwatch.stop("Lösungen bewertet (CPU): ");

	return _results;
}


__global__ void solutionEvaluatorKernel(CuPlans *plans, CuConstructionGraph *ptn, CuProblem *problem, CuEvaluationResult *result)
{
	int solutionId = blockIdx.x * blockDim.x + threadIdx.x;
	if (solutionId >= plans->getNumOfSolutions()) return;

	CuSolutionEvaluatorAco::analyseSolution(solutionId, plans, ptn, problem, result);
}


shared_ptr<CuEvaluationResult> CuSolutionEvaluatorAco::runOnGpu(shared_ptr <CuPlans> solutions)
{
	Stopwatch stopwatch;
	stopwatch.start();

	int blockSize; // Threads pro Block
	int minGridSize;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)solutionEvaluatorKernel));
	int numOfBlocks = (solutions->getNumOfSolutions() + blockSize - 1) / blockSize;
	dim3 dimGrid(numOfBlocks, 1);
	dim3 dimBlock(blockSize, 1);

	solutionEvaluatorKernel << <dimGrid, dimBlock >> > (solutions->getDevPtr(), _ptn->getDevPtr(), _problem->getDevPtr(), 
		_results->getDevPtr());

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	_results->copyToHost();

	stopwatch.stop("Lösungen bewertet (GPU): ");

	return _results;
}


shared_ptr<CuEvaluationResult> CuSolutionEvaluatorAco::run(shared_ptr <CuPlans> solutions)
{
	shared_ptr<CuEvaluationResult> retVal;

	switch (_plattform) {
	case PlattformConfig::UNDEFINED:
	{
		throw std::logic_error("Evaluator nicht konfiguriert!");
	}
	case PlattformConfig::CPU:
	{
		retVal = runOnCpu(solutions);
		break;
	}
	case PlattformConfig::GPU:
	{
		retVal = runOnGpu(solutions);
		//retVal = runOnCpu(solutions);
		break;
	}
	default:
		throw std::logic_error("Unbekannter Wert für PlattformConfig!");
	}

	return retVal;
}


CU_HSTDEV DurationInSeconds CuSolutionEvaluatorAco::calculateEmptyTrip(StopId from, StopId to, State &state, CuProblem *problem, CuVehicleType &vehicleType)
{
	DurationInSeconds retVal(0);
	EmptyTripId emptyTripId = problem->getConnectionMatrix().getEmptyTripId(from, to);
	if (emptyTripId.isValid()) {
		CuEmptyTrip emptyTrip = problem->getEmptyTrips().getEmptyTrip(emptyTripId);
		state.totalGrossCost += vehicleType.getDistanceDependentCosts(emptyTrip.distance);
		retVal = emptyTrip.duration;
	}
	return retVal;
}


CU_HSTDEV void CuSolutionEvaluatorAco::analyseSolution(int solutionId, CuPlans *plans, CuConstructionGraph *ptn, CuProblem *problem, CuEvaluationResult *result)
{
	AmountOfMoney netCostSum(0);
	AmountOfMoney grossCostSum(0);

	result->initialize(solutionId);

	int numOfCirculations = plans->getNumOfCirculations(solutionId);

	// Kosten des neuen Umlaufs berechnen und speichern.
	for (CirculationId circId(0); circId < numOfCirculations; circId++) {
		CuSolutionEvaluatorAco::analyseCirculation(solutionId, circId, plans, ptn, problem, result, netCostSum, grossCostSum);
	}

	result->setTotalCost(solutionId, AmountOfMoney(grossCostSum));
}


CU_HSTDEV void CuSolutionEvaluatorAco::analyseCirculation(int solutionId, CirculationId circulationId, CuPlans *plans, CuConstructionGraph *ptn, CuProblem *problem, CuEvaluationResult *result, AmountOfMoney &netCostSum, AmountOfMoney &grossCostSum)
{
	State state;

	StopId depotId = plans->getDepotId(solutionId, circulationId);
	VehicleTypeId vehicleTypeId = plans->getVehicleTypeId(solutionId, circulationId);
	CuVehicleType vehicleType = problem->getVehicleTypes().getVehicleType(vehicleTypeId);

	result->addCirculation(solutionId);

	state.currStop = depotId;

	int numOfSteps = plans->getNumOfNodes(solutionId, circulationId);
	for (CircStepIndex stepId(0); stepId < numOfSteps; stepId++) {
		NodeId nodeId = plans->getNodeId(solutionId, circulationId, stepId);
		EdgeId edgeId = plans->getEdgeId(solutionId, circulationId, stepId);
		analyseCirculationStep(nodeId, edgeId, state, ptn, problem, vehicleType);
	}

	// Beendet den aktuellen Umlauf (der Fahrer fährt zurück ins Depot).
	// Beinhaltet also auch die Einrückfahrt, falls diese nicht schon vorher berücksichtigt wurde.
	DurationInSeconds etDuration = calculateEmptyTrip(state.currStop, depotId, state, problem, vehicleType);

	// Kosten für das abschließende Aufladen im Depot verbuchen.
	// TODO Beim Lösen eines konventionellen VSP dürfen die Ladekosten nicht berechnet werden! rechargingCost und rechargingTime dazu auf 0 setzen!
	state.totalGrossCost = state.totalGrossCost + (int)vehicleType.rechargingCost;
	state.currentTime = state.currentTime + (int)vehicleType.rechargingTime;

	// Zeitabhängige Fahrzeugkosten für den gesamten Umlauf zu den Bruttokosten hinzu addieren.
	DurationInSeconds totalTime = DurationInSeconds((int)state.currentTime - (int)state.circulationStartTime);
	AmountOfMoney totalHourCost = vehicleType.getTimeDependentCosts(totalTime); // Zeitdifferenz in Sekunden und Kosten pro Stunde.
	state.totalGrossCost = state.totalGrossCost + totalHourCost;

	// Anschaffungskosten des Fahrzeugs berücksichtigen.
	state.totalGrossCost = state.totalGrossCost + vehicleType.vehCost;

	assert(state.totalGrossCost > 0);
	assert(state.totalNetCost > 0);

	float costRatio = (float)state.totalGrossCost / (float)state.totalNetCost;

	result->setCircStartTime(solutionId, circulationId, state.circulationStartTime);
	result->setCircCost(solutionId, circulationId, state.totalGrossCost);
	result->setCircCostRatio(solutionId, circulationId, costRatio);

	netCostSum = netCostSum + state.totalNetCost;
	grossCostSum = grossCostSum + state.totalGrossCost;
}


CU_HSTDEV void CuSolutionEvaluatorAco::analyseCirculationStep(NodeId nodeId, EdgeId edgeId, State &state, CuConstructionGraph *ptn, CuProblem *problem, CuVehicleType &vehicleType)
{
	CuNodeType nodeType = ptn->nodes.getNodeType(nodeId);

	switch (nodeType) {

	case CuNodeType::VehTypeDepotNode: 
	{
		StopId depot = StopId(ptn->nodes.getPayloadId(nodeId));
		DurationInSeconds etDuration = calculateEmptyTrip(state.currStop, depot, state, problem, vehicleType);
		state.currentTime = PointInTime((int)state.currentTime + (int)etDuration);
		state.currStop = depot;
	}
	break;

	case CuNodeType::ServiceTripNode:
	{
		/// Führt eine Servicefahrt durch. Beinhaltet auch eine evtl. nötige Leerfahrt.

		ServiceTripId serviceTripId = ServiceTripId(ptn->nodes.getPayloadId(nodeId));
		assert(serviceTripId.isValid());
		const CuServiceTrip &serviceTrip = problem->getServiceTrips().getServiceTrip(serviceTripId);

		DurationInSeconds etDuration = calculateEmptyTrip(state.currStop, serviceTrip.fromStopId, state, problem, vehicleType);

		if (!state.circulationStartTime.isValid()) {
			// Dies ist die erste SF im Umlauf: Startzeitpunkt festhalten.
			state.circulationStartTime = PointInTime((int)serviceTrip.departure - (int)etDuration);
		}

		state.totalGrossCost = state.totalGrossCost + vehicleType.getDistanceDependentCosts(serviceTrip.distance);
		state.totalNetCost = state.totalNetCost + vehicleType.getTimeDependentCosts(serviceTrip.getDuration());
		state.totalNetCost = state.totalNetCost + vehicleType.getDistanceDependentCosts(serviceTrip.distance);
		state.currentTime = serviceTrip.arrival;
		state.currStop = serviceTrip.toStopId;
	}
	break;

	case CuNodeType::ChargingStationNode:
	{
		/// Führt eine Aufladung durch.
		/// Beinhaltet auch eine ggf. nötige Verbindungsfahrt zur Ladestation.
		StopId csId = StopId(ptn->nodes.getPayloadId(nodeId));
		DurationInSeconds etDuration = calculateEmptyTrip(state.currStop, csId, state, problem, vehicleType);

		state.totalGrossCost = state.totalGrossCost + (int)vehicleType.rechargingCost;
		state.currentTime = PointInTime((int)state.currentTime + (int)vehicleType.rechargingTime + (int)etDuration);
		state.currStop = csId;
	}
	break;

	default:
		assert(false);
	}
}