#pragma once

#include <limits.h>
#include <memory>
#include <assert.h>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "Solution.h"
#include "SolutionKeyData.h"

using namespace std;

class BestSolutionManager {
public:
	BestSolutionManager();
	BestSolutionManager(shared_ptr<Solution> theSolution, AmountOfMoney theTotalCost, int theNumOfVehicles);
	~BestSolutionManager();

	bool checkSolution(const Solution &solution, SolutionKeyData solutionKeyData);

	const Solution& getBestSolution() const { assert(_bestSolution); return *_bestSolution; }
	int getNumOfVehicles() const { assert(_numOfVehicles >= 0); return _numOfVehicles; }
	AmountOfMoney getTotalCost() const { assert(_totalCost >= 0); return _totalCost; }

private:
	shared_ptr<Solution> _bestSolution;
	int _numOfVehicles;
	AmountOfMoney _totalCost;
};

