#include "CuSolutionGeneratorAcoCpu.h"

#include <thread>
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/StopWatch.h"
#include "CreatePlan.hpp"
#include "CuBoolVector1.hpp"
#include "RandomGpu.h"
#include "AntsRaceKernel.h"


CuSolutionGeneratorAcoCpu::CuSolutionGeneratorAcoCpu(std::shared_ptr<CuProblem> problem, std::shared_ptr<CuConstructionGraph> ptn, int populationSize, int maxNumOfCirculations, int maxNumOfNodes, int numOfThreads, float chargeLevel, bool verbose, bool keepBestSolution)
	: _numOfThreads(numOfThreads), CuSolutionGenerator(problem, ptn, populationSize, maxNumOfCirculations, maxNumOfNodes, verbose, keepBestSolution),
	_chargeLevel(chargeLevel)
{
}


CuSolutionGeneratorAcoCpu::~CuSolutionGeneratorAcoCpu()
{
}


shared_ptr<CuPlans> CuSolutionGeneratorAcoCpu::run(shared_ptr<CuSelectionResult> selectionResult, float chanceOfRandomSelection)
{
	Stopwatch stopwatch;
	stopwatch.start();

	if (_populationSize < _numOfThreads * 10) _numOfThreads = _populationSize / 10; // TODO Grenzwert konfigurierbar machen.

	if (_numOfThreads > 1) {
		int solutionsPerThead = (_populationSize + _numOfThreads - 1) / _numOfThreads;

		thread *t = new thread[_numOfThreads];

		int fromSolId, toSolId;

		for (int i = 0; i < _numOfThreads; ++i) {
			fromSolId = i * solutionsPerThead;
			toSolId = (i + 1) * solutionsPerThead - 1;
			if (toSolId >= _populationSize) toSolId = _populationSize - 1;
			t[i] = thread(antsRace, fromSolId, toSolId, selectionResult, _ptn, _problem, _plans, _batteryConsumption, _chargeLevel, _verbose, chanceOfRandomSelection, _keepBestSolution);
		}

		for (int i = 0; i < _numOfThreads; ++i) {
			t[i].join();
		}
	}
	else {
		antsRace(0, _populationSize - 1, selectionResult, _ptn, _problem, _plans, _batteryConsumption, _chargeLevel, _verbose, chanceOfRandomSelection, _keepBestSolution);
	}

	selectionResult->markAllAsAlive();
	selectionResult->copyToDevice();

	stopwatch.stop("Lösungen generiert (CPU): ");

	return _plans;
}


void CuSolutionGeneratorAcoCpu::antsRace(int fromSolId, int toSolId, shared_ptr<CuSelectionResult> selectionResult, shared_ptr<CuConstructionGraph> ptn,
	shared_ptr<CuProblem> problem, shared_ptr<CuPlans> solutions, shared_ptr<ConsumptionMatrix> consumption, float chargeLevel,
	bool verbose, float chanceOfRandomSelection, bool keepBestSolution)
{
	CuVector1<EdgeId> *activeEdges = new CuVector1<EdgeId>(Max_EdgesPerNode);
	CuVector1<float> *weights = new CuVector1<float>(Max_EdgesPerNode);
	CuVector1<bool> *activityStateOfNodes = new CuVector1<bool>(ptn->nodes.getNumOfNodes());
	int currentCirculationId = -1;

	for (int solutionId = fromSolId; solutionId <= toSolId; solutionId++) {
		activeEdges->initialize();
		weights->initialize();
		activityStateOfNodes->setAll(true);

		RandomCpu *rand = new RandomCpu();

		if (keepBestSolution && selectionResult->isAlive(solutionId)) {
			continue; // Nur inaktive ("tote") Lösungen werden ersetzt.
		}

		solutions->reInit(solutionId);

		AntsRaceState *state = new AntsRaceState(problem.get(), consumption.get(), chargeLevel, verbose);
		int numOfActiveEdges = 0;	// Anzahl der aktiven Kanten.
		float totalWeight = 0.0f;
		int numOfRemainingServTrips = problem->getServiceTrips().getNumOfServiceTrips();

		NodeId currNodeId = NodeId(0); // Id des aktuellen Knotens. Es beginnt mit dem Wurzelknoten.
		NodeId prevNodeId; // Id des vorhergehenden Knotens.
		NodeId targetNode;

		EmptyTripId emptyTripId = EmptyTripId::invalid(); // Id der am ausgewählten Knoten gespeicherten Verbindungsfahrt.

		createPlanCpu(solutionId, solutions.get(), ptn.get(), state, activeEdges, weights, activityStateOfNodes, numOfRemainingServTrips,
			rand, verbose, chanceOfRandomSelection);

		if (numOfRemainingServTrips > 0) {
			printf("Fehler: %i Servicefahrten konnten nicht eingeplant werden! Offenbar gibt es Servicefahrten, bei denen keine Ausrückfahrt möglich ist.\n", numOfRemainingServTrips);
			exit(-1);
		}

		delete rand;
		delete state;
	}
	delete activityStateOfNodes;
	delete activeEdges;
	delete weights;
}
