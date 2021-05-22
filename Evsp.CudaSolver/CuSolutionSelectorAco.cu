#include "CuSolutionSelectorAco.h"

#include <memory>
#include <exception>
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/StopWatch.h"
#include "CuSolutionEvaluatorAco.h"
#include "CuSolutionComparatorKernel.h"


using namespace std;


CuSolutionSelectorAco::CuSolutionSelectorAco(shared_ptr<CuConstructionGraph> ptn, bool dumpBestSolution, bool dumpWorstSolution, PlattformConfig plattform, int populationSize, int maxNumOfCirculations, int maxNumOfNodes)
	: _ptn(ptn), _dumpBestSolution(dumpBestSolution), _dumpWorstSolution(dumpWorstSolution), _plattform(plattform),  
	_rand(new RandomCpu()), 
	_selectionResult(new CuSelectionResult(populationSize, maxNumOfCirculations, maxNumOfNodes))
{
	
}


CuSolutionSelectorAco::~CuSolutionSelectorAco()
{
}


shared_ptr<CuSelectionResult> CuSolutionSelectorAco::run(shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuComparatorResult> compResult, shared_ptr<CuPlans> solutions)
{
	shared_ptr<CuSelectionResult> retVal;

	switch (_plattform) {
	case PlattformConfig::UNDEFINED:
	{
		throw std::logic_error("Selector nicht konfiguriert!");
	}
	case PlattformConfig::CPU:
	{
		retVal = runOnCpu(evalResult, compResult, solutions);
		break;
	}
	case PlattformConfig::GPU:
	{
		retVal = runOnGpu(evalResult, compResult, solutions);
		break;
	}
	default:
		throw std::logic_error("Unbekannter Wert für PlattformConfig!");
	}

	return retVal;
}


shared_ptr<CuSelectionResult> CuSolutionSelectorAco::runOnCpu(shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuComparatorResult> compResult, shared_ptr<CuPlans> solutions)
{
	Stopwatch stopwatch;
	stopwatch.start();

	_selectionResult->markAllAsDead();
	
	SolutionKeyData toBeChecked = compResult->getLowestTotalCost();
	bool newBestInClass = _selectionResult->checkTheBestSolutionInClass(toBeChecked);

	toBeChecked = compResult->getHighestTotalCost();
	_selectionResult->checkTheWorstSolutionInClass(toBeChecked);

	_selectionResult->copyToDevice();

	stopwatch.stop("Lösung selektiert (CPU): ");

	if (_dumpBestSolution) {
		cout << endl << "Beste Lösung des letzten Durchlaufs:" << endl;
		solutions->dump(_ptn, _selectionResult->getBestSolution().getSolutionId());
	}

	if (_dumpWorstSolution) {
		cout << endl << "Schlechteste Lösung des letzten Durchlaufs:" << endl;
		solutions->dump(_ptn, _selectionResult->getWorstSolution().getSolutionId());
	}

	return _selectionResult;
}


shared_ptr<CuSelectionResult> CuSolutionSelectorAco::runOnGpu(shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuComparatorResult> compResult, shared_ptr<CuPlans> solutions)
{
	return runOnCpu(evalResult, compResult, solutions);
}


