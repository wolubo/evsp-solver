#include "CuSolutionComparator.h"
#include <tuple>
#include <assert.h>
#include "CuSolutionComparatorKernel.h"
#include "EVSP.BaseClasses/StopWatch.h"



CuSolutionComparator::CuSolutionComparator(int populationSize, PlattformConfig plattform)
	: _populationSize(populationSize), _comparatorResults(new CuComparatorResult()), _plattform(plattform)
{
}


CuSolutionComparator::~CuSolutionComparator()
{
}


shared_ptr<CuComparatorResult> CuSolutionComparator::run(shared_ptr<CuEvaluationResult> evaluationResult)
{
	shared_ptr<CuComparatorResult> retVal;

	switch (_plattform) {
	case PlattformConfig::UNDEFINED:
	{
		throw std::logic_error("Comparator nicht konfiguriert!");
	}
	case PlattformConfig::CPU:
	{
		retVal = runOnCpu(evaluationResult);
		break;
	}
	case PlattformConfig::GPU:
	{
		retVal = runOnGpu(evaluationResult);
		retVal = runOnCpu(evaluationResult);
		break;
	}
	default:
		throw std::logic_error("Unbekannter Wert für PlattformConfig!");
	}

	return retVal;
}


bool CuSolutionComparator::check(const CuComparatorResult &comparatorResult, const CuEvaluationResult &evaluationResult)
{
	int evalTotalCost;
	int compTotalCost = (int)comparatorResult.getLowestTotalCost().getTotalCost();
	for (int i = 0; i < evaluationResult.getNumOfSolutions(); i++) {
		evalTotalCost = (int)evaluationResult.getTotalCost(i);
		if (evalTotalCost < compTotalCost) {
			cerr << "Der Comparator findet die Lösung mit den geringsten Gesamtkosten nicht!" << endl;
			return false;
		}
	}

	evalTotalCost = (int)evaluationResult.getTotalCost(comparatorResult.getLowestTotalCost().getSolutionId());
	if (evalTotalCost != compTotalCost) {
		cerr << "Der Comparator hat die Id der Lösung mit den geringsten Gesamtkosten nicht korrekt gespeichert!" << endl;
		return false;
	}

	return true;
}


shared_ptr<CuComparatorResult> CuSolutionComparator::runOnCpu(shared_ptr<CuEvaluationResult> evaluationResult)
{
	Stopwatch stopwatch;
	stopwatch.start();

	AmountOfMoney currentTotalCost;

	SolutionKeyData lowestTotalCost(-1, AmountOfMoney(INT_MAX), INT_MAX);
	SolutionKeyData highestTotalCost(-1, AmountOfMoney(0), 0);
	SolutionKeyData lowestNumOfVehicles(-1, AmountOfMoney(INT_MAX), INT_MAX);
	SolutionKeyData highestNumOfVehicles(-1, AmountOfMoney(0), 0);
	CirculationKeyData lowestCircCostRatio(-1, CirculationId::invalid(), FLT_MAX);
	CirculationKeyData highestCircCostRatio(-1, CirculationId::invalid(), 0.0f);

	for (int solutionId = 0; solutionId < _populationSize; solutionId++) { // TODO Schleife parallelisieren.

		currentTotalCost = evaluationResult->getTotalCost(solutionId);
		assert(currentTotalCost > 0);

		int currentNumOfVehicles = evaluationResult->getNumOfCirculations(solutionId);
		assert(currentNumOfVehicles > 0);

		if (currentTotalCost < lowestTotalCost.getTotalCost()) {
			lowestTotalCost = SolutionKeyData(solutionId, currentTotalCost, currentNumOfVehicles);
		}

		if (currentTotalCost > highestTotalCost.getTotalCost()) {
			highestTotalCost = SolutionKeyData(solutionId, currentTotalCost, currentNumOfVehicles);
		}

		if (currentNumOfVehicles < lowestNumOfVehicles.getNumOfCirculations()) {
			lowestNumOfVehicles = SolutionKeyData(solutionId, currentTotalCost, currentNumOfVehicles);
		}
		else if (currentNumOfVehicles == lowestNumOfVehicles.getNumOfCirculations()) {
			if (currentTotalCost < lowestNumOfVehicles.getTotalCost()) {
				lowestNumOfVehicles = SolutionKeyData(solutionId, currentTotalCost, currentNumOfVehicles);
			}
		}

		if (currentNumOfVehicles > highestNumOfVehicles.getNumOfCirculations()) {
			highestNumOfVehicles = SolutionKeyData(solutionId, currentTotalCost, currentNumOfVehicles);
		}
		else if (currentNumOfVehicles == highestNumOfVehicles.getNumOfCirculations()) {
			if (currentTotalCost > highestNumOfVehicles.getTotalCost()) {
				highestNumOfVehicles = SolutionKeyData(solutionId, currentTotalCost, currentNumOfVehicles);
			}
		}

		for (CirculationId circulationId(0); circulationId < evaluationResult->getNumOfCirculations(solutionId); circulationId++) {

			float currentCircCostRatio = evaluationResult->getCircCostRatio(solutionId, circulationId);
			assert(currentCircCostRatio >= 1.0f);

			if (currentCircCostRatio < lowestCircCostRatio.value) {
				lowestCircCostRatio = CirculationKeyData(solutionId, circulationId, currentCircCostRatio);
			}

			if (currentCircCostRatio > highestCircCostRatio.value) {
				highestCircCostRatio = CirculationKeyData(solutionId, circulationId, currentCircCostRatio);
			}

		}
	}

	assert(lowestTotalCost.getSolutionId() != -1);
	assert(lowestTotalCost.getTotalCost() != INT_MAX);
	_comparatorResults->_lowestTotalCost = lowestTotalCost;

	assert(highestTotalCost.getSolutionId() != -1);
	assert(highestTotalCost.getTotalCost() != 0);
	_comparatorResults->_highestTotalCost = highestTotalCost;

	assert(lowestNumOfVehicles.getSolutionId() != -1);
	assert(lowestNumOfVehicles.getNumOfCirculations() != INT_MAX);
	_comparatorResults->_lowestNumOfVehicles = lowestNumOfVehicles;

	assert(highestNumOfVehicles.getSolutionId() != -1);
	assert(highestNumOfVehicles.getNumOfCirculations() != 0);
	_comparatorResults->_highestNumOfVehicles = highestNumOfVehicles;

	assert(lowestCircCostRatio.solutionId != -1);
	assert(lowestCircCostRatio.circulationId.isValid());
	assert(lowestCircCostRatio.value != FLT_MAX);
	_comparatorResults->_lowestCircCostRatio = lowestCircCostRatio;

	assert(highestCircCostRatio.solutionId != -1);
	assert(highestCircCostRatio.circulationId.isValid());
	assert(highestCircCostRatio.value != 0.0f);
	_comparatorResults->_highestCircCostRatio = highestCircCostRatio;

	_comparatorResults->copyToDevice();

	stopwatch.stop("Lösungen verglichen (CPU): ");

	return _comparatorResults;
}


shared_ptr<CuComparatorResult> CuSolutionComparator::runOnGpu(shared_ptr<CuEvaluationResult> evaluationResult)
{
	Stopwatch stopwatch;
	stopwatch.start();

	startSolutionComparatorKernel(_populationSize, evaluationResult.get(), _comparatorResults.get());

	_comparatorResults->copyToHost();

	stopwatch.stop("Lösungen verglichen (GPU): ");

	return _comparatorResults;
}


