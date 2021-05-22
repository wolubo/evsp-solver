#include "BestSolutionManager.h"



BestSolutionManager::BestSolutionManager()
	: _bestSolution(0), _numOfVehicles(INT_MAX), _totalCost(INT_MAX)
{
}


BestSolutionManager::BestSolutionManager(shared_ptr<Solution> theSolution, AmountOfMoney theTotalCost, int theNumOfVehicles)
	: _bestSolution(theSolution), _numOfVehicles(theNumOfVehicles), _totalCost(theTotalCost)
{
}


BestSolutionManager::~BestSolutionManager()
{
}


bool BestSolutionManager::checkSolution(const Solution &solution, SolutionKeyData solutionKeyData)
{
	bool foundNewBest = false;

	int solNumOfVehicles = solutionKeyData.getNumOfCirculations();
	AmountOfMoney solTotalCost = solutionKeyData.getTotalCost();

	if (!_bestSolution) {
		foundNewBest = true;
	}
	else {
		foundNewBest = _totalCost > solTotalCost;
	}

	if (foundNewBest) {
		_bestSolution = make_shared<Solution>(solution);
		_numOfVehicles = solNumOfVehicles;
		_totalCost = solTotalCost;
	}

	return foundNewBest;
}
