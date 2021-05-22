#include "CuSaSolver.h"
#include <iomanip>


CuSaSolver::CuSaSolver(std::shared_ptr<CuProblem> problem, PlattformConfig plattform, int numOfThreads, SaParams params,
	shared_ptr<ResultLogger> resultLogger)
	: CuSolver(resultLogger), _problem(problem), _params(params), _solutions(),
	_bestSolutionManager(), _numOfThreads(numOfThreads), _temperature(params.startTemperature, params.minTemperature)
{
}


CuSaSolver::~CuSaSolver()
{
}


shared_ptr<CuSolution> CuSaSolver::getSolution()
{
	assert(false);
	return 0;
}


void CuSaSolver::loop()
{
	if (_params.printStats) {
		cout << "Temperatur=" << setprecision(1) << (float)_temperature << ", ";
	}
	_solutions->mutateSolutions(_params, _numOfThreads, _problem, _temperature);
	findBestSolution(false);
	_temperature *= (1.0f - _params.coolingRate);
}


void CuSaSolver::setup()
{
	_solutions = make_shared<Solutions>(_numOfThreads, _problem, _params);
	findBestSolution(true);
}


void CuSaSolver::findBestSolution(bool printStartSolution)
{
	SolutionKeyData bestSolutionKeyData = _solutions->getBestSolution(_problem);
	const Solution& bestSolution = _solutions->getSolution(bestSolutionKeyData.getSolutionId());
	bool newBestSolution = _bestSolutionManager.checkSolution(bestSolution, bestSolutionKeyData);
	if (newBestSolution) {
		if (printStartSolution) {
			addResult("STARTLÖSUNG", _bestSolutionManager.getTotalCost(), _bestSolutionManager.getNumOfVehicles());
		}
		else {
			addResult("NEUE LÖSUNG", _bestSolutionManager.getTotalCost(), _bestSolutionManager.getNumOfVehicles());
		}
	}
}


void CuSaSolver::teardown()
{

}


