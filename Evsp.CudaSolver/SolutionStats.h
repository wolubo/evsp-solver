#pragma once

#include <assert.h>
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuProblem.h"

using namespace std;

class Solution;

class SolutionStats
{
public:
	SolutionStats();
	SolutionStats(int numOfCirculations, AmountOfMoney totalCost, float bestCircCostRatio, float worstCircCostRatio, int bestTripLenght, int worstTripLenght, int lowestNumOfActions, int _highestNumOfActions);
	SolutionStats(const SolutionStats &other);
	~SolutionStats();

	SolutionStats& operator=(const SolutionStats &rhs);
	bool operator==(const SolutionStats &rhs);
	bool operator!=(const SolutionStats &rhs);

	int getNumOfCirculations() const { return _numOfCirculations; }
	AmountOfMoney getTotalCost() const { return _totalCost; }
	float getBestCircCostRatio() const { return _bestCircCostRatio; }
	float getWorstCircCostRatio() const { return _worstCircCostRatio; }
	int getBestTripLenght() const { return _bestTripLenght; }
	int getWorstTripLenght() const { return _worstTripLenght; }
	int getLowestNumOfActions()const { return _lowestNumOfActions; }
	int getHighestNumOfActions()const { return _highestNumOfActions; }

protected:
	int _numOfCirculations;
	AmountOfMoney _totalCost;
	float _bestCircCostRatio;
	float _worstCircCostRatio;
	int _bestTripLenght;
	int _worstTripLenght;
	int _lowestNumOfActions;
	int _highestNumOfActions;
};

