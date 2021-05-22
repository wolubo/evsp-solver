#include "SolutionStats.h"
#include "Solution.h"
#include "CirculationStats.h"

SolutionStats::SolutionStats()
	: _numOfCirculations(-1), _totalCost(), _bestCircCostRatio(FLT_MAX), _worstCircCostRatio(0.0f), _bestTripLenght(0),	_worstTripLenght(INT_MAX), _lowestNumOfActions(INT_MAX), _highestNumOfActions(0)

{
}


SolutionStats::SolutionStats(int numOfCirculations, AmountOfMoney totalCost, float bestCircCostRatio, float worstCircCostRatio, int bestTripLenght, int worstTripLenght, int lowestNumOfActions, int highestNumOfActions)
	: _numOfCirculations(numOfCirculations), _totalCost(totalCost), _bestCircCostRatio(bestCircCostRatio), _worstCircCostRatio(worstCircCostRatio), _bestTripLenght(bestTripLenght), _worstTripLenght(worstTripLenght), _lowestNumOfActions(lowestNumOfActions), _highestNumOfActions(highestNumOfActions)
{
}


SolutionStats::SolutionStats(const SolutionStats &other)
	: _numOfCirculations(other._numOfCirculations), _totalCost(other._totalCost),
	_bestCircCostRatio(other._bestCircCostRatio), _worstCircCostRatio(other._worstCircCostRatio),
	_bestTripLenght(other._bestTripLenght), _worstTripLenght(other._worstTripLenght), _lowestNumOfActions(other._lowestNumOfActions), _highestNumOfActions(other._highestNumOfActions)
{
}


SolutionStats::~SolutionStats()
{
}


SolutionStats& SolutionStats::operator=(const SolutionStats &rhs)
{
	if (this != &rhs) {
		_numOfCirculations = rhs._numOfCirculations;
		_totalCost = rhs._totalCost;
		_bestCircCostRatio = rhs._bestCircCostRatio;
		_worstCircCostRatio = rhs._worstCircCostRatio;
		_bestTripLenght = rhs._bestTripLenght;
		_worstTripLenght = rhs._worstTripLenght;
	}
	return *this;
}


bool SolutionStats::operator==(const SolutionStats &rhs)
{
	if (this == &rhs) return true;

	if (_numOfCirculations != rhs._numOfCirculations) return false;
	if (_totalCost != rhs._totalCost) return false;
	if (_bestCircCostRatio != rhs._bestCircCostRatio) return false;
	if (_worstCircCostRatio != rhs._worstCircCostRatio) return false;
	if (_bestTripLenght != rhs._bestTripLenght) return false;
	if (_worstTripLenght != rhs._worstTripLenght) return false;

	return true;
}


bool SolutionStats::operator!=(const SolutionStats &rhs)
{
	return !(*this == rhs);
}
