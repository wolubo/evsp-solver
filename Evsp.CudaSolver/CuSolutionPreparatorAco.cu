#include "CuSolutionPreparatorAco.h"
#include <thread>
#include <assert.h>
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/StopWatch.h"
#include "CuSolutionPreparatorAcoKernels.h"

#pragma warning(disable: 4244)

CuSolutionPreparatorAco::CuSolutionPreparatorAco(int populationSize, shared_ptr<CuConstructionGraph> ptn, AcoQualifiers qualifiers, PlattformConfig plattform,
	int numOfThreads)
	: _populationSize(populationSize), _ptn(ptn), _acoQualifiers(qualifiers), _plattform(plattform), _numOfThreads(numOfThreads)
{
	int numOfNodes = _ptn->nodes.getNumOfNodes();
	int numOfEdges = _ptn->edges->getMaxNumOfOutEdges();
	_lock = new CuLockMatrix1(numOfNodes, numOfEdges);
}


CuSolutionPreparatorAco::~CuSolutionPreparatorAco()
{
	if (_lock) delete _lock;
}


void CuSolutionPreparatorAco::run(shared_ptr<CuPlans> solutions, shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuSelectionResult> selectResult, shared_ptr<CuComparatorResult> compResult, float fading, bool normalizeEdgeWeights)
{
	switch (_plattform) {
	case PlattformConfig::UNDEFINED:
	{
		throw std::logic_error("Preparator nicht konfiguriert!");
	}
	case PlattformConfig::CPU:
	{
		runOnCpu(solutions, evalResult, selectResult, compResult, fading, normalizeEdgeWeights);
		break;
	}
	case PlattformConfig::GPU:
	{
		runOnGpu(solutions, evalResult, selectResult, compResult, fading, normalizeEdgeWeights);
		break;
	}
	default:
		throw std::logic_error("Unbekannter Wert für PlattformConfig!");
	}
}


void CuSolutionPreparatorAco::performNormalizeEdgeWeights(shared_ptr<CuPlans> solutions) 
{
	for (int solId = 0; solId < _populationSize; solId++) {
		float highest = 0.0f;
		//float lowest = FLT_MAX;
		for (CirculationId c(0); c < solutions->getNumOfCirculations(solId); c++) {
			for (CircStepIndex s(0); s < solutions->getNumOfNodes(solId, c); s++) {
				NodeId nodeId = solutions->getNodeId(solId, c, s);
				EdgeId edgeId = solutions->getEdgeId(solId, c, s);
				if (edgeId.isValid()) {
					float current = _ptn->edges->getWeight(nodeId, edgeId);
					highest = std::fmaxf(highest, current);
					//lowest = std::fminf(lowest, current);
				}
			}
		}

		float initialWeight = _acoQualifiers.initialWeight * _populationSize;

		for (CirculationId c(0); c < solutions->getNumOfCirculations(solId); c++) {
			for (CircStepIndex s(0); s < solutions->getNumOfNodes(solId, c); s++) {
				NodeId nodeId = solutions->getNodeId(solId, c, s);
				EdgeId edgeId = solutions->getEdgeId(solId, c, s);
				if (edgeId.isValid()) {
					float current = _ptn->edges->getWeight(nodeId, edgeId);
					float factor = highest / initialWeight;
					float newValue = current / factor;
					if (newValue < 0.0f)
						newValue = 0.0f;
					else if (newValue > initialWeight)
						newValue = initialWeight;

					assert(isnormal(newValue));

					_ptn->edges->setWeight(nodeId, edgeId, newValue);
				}
			}
		}
	}
}


void CuSolutionPreparatorAco::runOnCpu(shared_ptr<CuPlans> solutions, shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuSelectionResult> selectResult, shared_ptr<CuComparatorResult> compResult, float fading, bool normalizeEdgeWeights)
{
	Stopwatch stopwatch;
	stopwatch.start();

	// Alte Spuren verblassen lassen.
	_ptn->edges->fadeTracks(fading);

	if (selectResult->hasNewBestSolution() && normalizeEdgeWeights) {
		// Es gibt einen neuen Bestwert. Deshalb alle Kantengewichte auf das Intervall [0, 1] normieren.
		performNormalizeEdgeWeights(solutions);
	}

	if (_populationSize < _numOfThreads * 10) _numOfThreads = 1; // TODO Grenzwert konfigurierbar machen.

	std::mutex oneMutex;

	if (_numOfThreads > 1) {
		int solutionsPerThead = (_populationSize + _numOfThreads - 1) / _numOfThreads;

		thread *t = new thread[_numOfThreads];

		int fromSolId, toSolId;

		for (int i = 0; i < _numOfThreads; ++i) {
			fromSolId = i * solutionsPerThead;
			toSolId = (i + 1) * solutionsPerThead - 1;
			if (toSolId >= _populationSize) toSolId = _populationSize - 1;
			t[i] = thread(updatePheromoneTracksOnCpu, fromSolId, toSolId, this, std::ref(oneMutex), solutions, evalResult, selectResult, compResult);
		}

		for (int i = 0; i < _numOfThreads; ++i) {
			t[i].join();
		}
	}
	else {
		updatePheromoneTracksOnCpu(0, _populationSize - 1, this, std::ref(oneMutex), solutions, evalResult, selectResult, compResult);
	}

	_ptn->edges->copyToDevice();

	stopwatch.stop("Kantengewichte neu berechnet (CPU): ");
}


void CuSolutionPreparatorAco::updatePheromoneTracksOnCpu(int fromSolId, int toSolId, CuSolutionPreparatorAco *p, std::mutex &theMutex, shared_ptr<CuPlans> solutions, shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuSelectionResult> selectResult, shared_ptr<CuComparatorResult> compResult)
{
	// Spuren der in der letzten Runde gelaufenen Ameisen berechnen und einsammeln
	for (int solNum = fromSolId; solNum <= toSolId; solNum++) {

		SolutionKeyData best = selectResult->getBestSolution();

		// Berechne den von den Gesamtkosten der aktuellen Lösung abhängigen Teil der Spur.
		double totalCostValue;
		double currentTC = (int)evalResult->getTotalCost(solNum);
		if (p->_populationSize > 1) {
			double highestTC, lowestTC;
			highestTC = (float)compResult->getHighestTotalCost().getTotalCost();
			//lowestTC = (float)compResult->getLowestTotalCost().getTotalCost();
			lowestTC = (float)best.getTotalCost();
			totalCostValue = (highestTC - currentTC) / (highestTC - lowestTC);
		}
		else {
			totalCostValue = 1.0 / currentTC;
		}
		if (totalCostValue < 0.0) totalCostValue = 0.0;
		if (totalCostValue > 1.0) totalCostValue = 1.0;

		// Berechne den von der Fahrzeuganzahl der aktuellen Lösung abhängigen Teil der Spur.
		double numOfVehiclesValue;
		double currentNumOfVehicles = evalResult->getNumOfCirculations(solNum);
		if (p->_populationSize > 1) {
			double highestNumOfVehicles, lowestNumOfVehicles;
			highestNumOfVehicles = (float)compResult->getHighestNumOfVehicles().getNumOfCirculations();
			lowestNumOfVehicles = (float)compResult->getLowestNumOfVehicles().getNumOfCirculations();
			numOfVehiclesValue = (highestNumOfVehicles - currentNumOfVehicles) / (highestNumOfVehicles - lowestNumOfVehicles);
		}
		else {
			numOfVehiclesValue = 1.0 / currentNumOfVehicles;
		}
		if (numOfVehiclesValue < 0.0) numOfVehiclesValue = 0.0;
		if (numOfVehiclesValue > 1.0) numOfVehiclesValue = 1.0;

		for (CirculationId c(0); c < solutions->getNumOfCirculations(solNum); c++) {

			// Berechne den vom Kostenverhältnis des Umlaufs abhängigen Teil der Spur.
			double currentCCR = (int)evalResult->getCircCostRatio(solNum, c);
			CirculationKeyData lowestCCR = compResult->getLowestCircCostRatio();
			CirculationKeyData highestCCR = compResult->getHighestCircCostRatio();
			double ccrValue = (highestCCR.value - currentCCR) / (highestCCR.value - lowestCCR.value);
			if (ccrValue < 0.0) ccrValue = 0.0;
			if (ccrValue > 1.0) ccrValue = 1.0;

			// Berechne das 1. Teilergebnis der neuen Spur.
			double newPheromoneTrackValue;
			newPheromoneTrackValue = p->_acoQualifiers.totalCostQualifier * totalCostValue;
			newPheromoneTrackValue += p->_acoQualifiers.numOfVehiclesQualifier * numOfVehiclesValue;
			newPheromoneTrackValue += p->_acoQualifiers.circCostRatioQualifier * ccrValue;

			// Normiere das 1. Teilergebnis auf das Intervall [0.0;1.0].
			//newPheromoneTrackValue /= (p->_acoQualifiers.totalCostQualifier + p->_acoQualifiers.numOfVehiclesQualifier + p->_acoQualifiers.circCostRatioQualifier);

			// Schwäche die schlechten Werte ab und betone die starken Werte.
			newPheromoneTrackValue = pow(newPheromoneTrackValue, p->_acoQualifiers.weakenAllBadSolutions);
			
			assert(newPheromoneTrackValue==0.0f || isnormal(newPheromoneTrackValue));

			// Aktualisiere die Spuren an den Kanten.
			for (CircStepIndex s(0); s < solutions->getNumOfNodes(solNum, c); s++) {
				NodeId nodeId = solutions->getNodeId(solNum, c, s);
				EdgeId edgeId = solutions->getEdgeId(solNum, c, s);
				if (edgeId.isValid()) {
					theMutex.lock(); // TODO Einen individuellen Mutex pro Node/Edge verwenden.
					p->_ptn->edges->addWeight(nodeId, edgeId, newPheromoneTrackValue); // Neue Spur setzen.
					theMutex.unlock();
				}
			}
		}
	}
}

void CuSolutionPreparatorAco::runOnGpu(shared_ptr<CuPlans> solutions, shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuSelectionResult> selectResult, shared_ptr<CuComparatorResult> compResult, float fading, bool normalizeEdgeWeights)
{
	runOnCpu(solutions, evalResult, selectResult, compResult, fading, normalizeEdgeWeights);

	//Stopwatch stopwatch;
	//stopwatch.start();

	//int blockSize_x, blockSize_y;

	//shared_ptr<CuMatrix1<float>> pheromonTrackSum = make_shared<CuMatrix1<float>>(_numOfNodes, _numOfEdges);
	//shared_ptr<CuMatrix1<int>> counters = make_shared<CuMatrix1<int>>(_numOfNodes, _numOfEdges);

	////_solutions->copyToHost(); // TODO: Code aufs Device verlagern, damit die Kopieroperation unnötig wird.
	////evalResult->copyToHost();
	////selectResult->copyToHost();

	//double avgWeight = 0.0; // Durchschnittliches Kantengewicht (für die noch nicht besuchten Kanten).

	//// Spuren der in der letzten Runde gelaufenen Ameisen berechnen und einsammeln
	//for (int solNum = 0; solNum < _populationSize; solNum++) {

	//	/* Gesamtkosten der aktuellen Lösung im Verhältnis zu den besten und den schlechtesten Gesamtkosten aller Lösungen. */
	//	double current = (int)evalResult->getTotalCost(solNum);
	//	double worst = (int)selectResult->worstSolutionTC;
	//	//float best = (int)selectResult->bestInClass_totalCost;
	//	double best = (int)selectResult->bestSolutionTC;
	//	double newValue;
	//	if (_populationSize > 1) {
	//		assert(worst - best > 0.0f);
	//		newValue = (worst - current) / (worst - best);
	//		newValue = pow(newValue, 3); // Schlechter Lösungen abschwächen.
	//		//if (current == best) {
	//			// Beste Lösung überbetonen.
	//			//newValue *= 2.0f;
	//		//}
	//		newValue /= _populationSize;
	//	}
	//	else {
	//		newValue = 1.0f / current;
	//	}

	//	//double newValue = 100000.0f;
	//	//newValue /= current;
	//	//newValue /= populationSize;
	//	//if (current == best) {
	//	//	// Beste Lösung überbetonen.
	//	//	newValue *= 2.0f;
	//	//}

	//	/* Verhältnis durchschnittliche Brutto-/Nettokosten der Umläufe */
	//	//double circCostRatio = evalResult->getAvgCircCostRatio(solNum);
	//	////circCostRatio *= circCostRatio; // Quadrieren, um Werte schlechter Lösungen abzuschwächen.
	//	//double newValue = 1.0 / circCostRatio; // Ergebnis ist immer <=1.0
	//	//newValue *= newValue; // Quadrieren, um Werte schlechter Lösungen abzuschwächen.

	//	avgWeight += newValue;

	//	// Alle Kanten des aktuellen Entscheidungspfads behandeln.
	//	int numOfCirculations = _solutions->getNumOfCirculations(solNum);
	//	int numOfSteps = _solutions->getMaxNumOfActions();
	//	assert(numOfSteps < 256);
	//	blockSize_x = 1024 / numOfSteps;
	//	blockSize_y = numOfSteps;
	//	int numOfBlocks_x = (numOfCirculations + blockSize_x - 1) / blockSize_x;
	//	int numOfBlocks_y = (numOfSteps + blockSize_y - 1) / blockSize_y;
	//	dim3 dimGrid(numOfBlocks_x, numOfBlocks_y);
	//	dim3 dimBlock(blockSize_x, blockSize_y);
	//	collectPheromonesKernel<<<dimGrid, dimBlock>>>(solNum, numOfCirculations, numOfSteps, pheromonTrackSum->getDevPtr(), counters->getDevPtr(), _solutions->getDevPtr(), newValue, _lock->getDevPtr());
	//	CUDA_CHECK(cudaGetLastError());
	//}
	//CUDA_CHECK(cudaDeviceSynchronize());

	//avgWeight /= _populationSize;

	//// Kantengewichte ändern
	//blockSize_x = 32;
	//blockSize_y = 32;
	//int numOfBlocks_x = (_numOfNodes + blockSize_x - 1) / blockSize_x;
	//int numOfBlocks_y = (_numOfEdges + blockSize_y - 1) / blockSize_y;
	//dim3 dimGrid(numOfBlocks_x, numOfBlocks_y);
	//dim3 dimBlock(blockSize_x, blockSize_y);
	//updatePheromoneTracksKernel<<<dimGrid, dimBlock>>>(pheromonTrackSum->getDevPtr(), counters->getDevPtr(),
	//	_ptn->edges->getDevPtr(), _fading, avgWeight, selectResult->newBestSolution, _numOfNodes);
	//CUDA_CHECK(cudaGetLastError());
	//CUDA_CHECK(cudaDeviceSynchronize());
	//
	//_ptn->edges->copyToHost();

	//stopwatch.stop("Kantengewichte neu berechnet (GPU): ");

	////_ptn->dumpDecisionNet(_problem);
}

