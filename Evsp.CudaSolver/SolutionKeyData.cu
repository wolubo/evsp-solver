#include "SolutionKeyData.h"
#include <string>


CU_HSTDEV SolutionKeyData::SolutionKeyData()
	: _solutionId(-1), _numOfCirculations(-1), _totalCost()
{
}


CU_HSTDEV SolutionKeyData::SolutionKeyData(const SolutionKeyData &other)
	: _solutionId(other._solutionId), _numOfCirculations(other._numOfCirculations), _totalCost(other._totalCost)
{
}


CU_HSTDEV SolutionKeyData::SolutionKeyData(int theSolutionId, AmountOfMoney totalCost, int numOfCirculations)
	: _solutionId(theSolutionId), _numOfCirculations(numOfCirculations), _totalCost(totalCost)
{
}


SolutionKeyData& SolutionKeyData::operator=(const SolutionKeyData &rhs)
{
	if (this != &rhs) {
		_solutionId = rhs._solutionId;
		_numOfCirculations = rhs._numOfCirculations;
		_totalCost = rhs._totalCost;
	}
	return *this;
}


bool SolutionKeyData::operator==(const SolutionKeyData &rhs)
{
	if (this == &rhs) return true;

	if (_solutionId != rhs._solutionId) return false;
	if (_numOfCirculations != rhs._numOfCirculations) return false;
	if (_totalCost != rhs._totalCost) return false;

	return true;
}


bool SolutionKeyData::operator!=(const SolutionKeyData &rhs)
{
	return !(*this == rhs);
}


std::string SolutionKeyData::toString()
{
	return "solutionId=" + std::to_string(_solutionId) 
		+ ", numOfCirculations=" + std::to_string(_numOfCirculations)
		+ ", totalCost=" + std::to_string((int)_totalCost);
}
