#pragma once

#include <assert.h>
#include "EVSP.BaseClasses/Typedefs.h"
#include "Solution.h"

class SolutionKeyData 
{
public:
	CU_HSTDEV SolutionKeyData();

	CU_HSTDEV SolutionKeyData(const SolutionKeyData &other);

	CU_HSTDEV SolutionKeyData(int theSolutionId, AmountOfMoney totalCost, int numOfCirculations);

	CU_HSTDEV SolutionKeyData& operator=(const SolutionKeyData &rhs);

	CU_HSTDEV bool operator==(const SolutionKeyData &rhs);

	CU_HSTDEV bool operator!=(const SolutionKeyData &rhs);

	std::string toString();

	CU_HSTDEV int getSolutionId() const { return _solutionId; }
	CU_HSTDEV int getNumOfCirculations() const { return _numOfCirculations; }
	CU_HSTDEV AmountOfMoney getTotalCost() const { return _totalCost; }

private:
	int _solutionId;
	int _numOfCirculations;
	AmountOfMoney _totalCost;
};

