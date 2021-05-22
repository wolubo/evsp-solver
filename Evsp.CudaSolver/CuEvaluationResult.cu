#include "CuEvaluationResult.h"

#include "cuda_runtime.h"



CuEvaluationResult::CuEvaluationResult(int numOfSolutions, int maxNumOfCirculations)
	: _devicePtr(0), _numOfSolutions(numOfSolutions)
{
	_numOfCirculations = new CuVector1<int>(numOfSolutions);
	_totalCost = new CuVector1<AmountOfMoney>(numOfSolutions);
	_circStartTime = new CuMatrix1<PointInTime>(numOfSolutions, maxNumOfCirculations);
	_circCost = new CuMatrix1<AmountOfMoney>(numOfSolutions, maxNumOfCirculations);
	_circCostRatio = new CuMatrix1<float>(numOfSolutions, maxNumOfCirculations);
}


CuEvaluationResult::CuEvaluationResult(const CuEvaluationResult& other)
	: _devicePtr(0), _numOfSolutions(other._numOfSolutions), _numOfCirculations(0), _totalCost(0), _circStartTime(0), 
	_circCost(0), _circCostRatio(0)
{
	if (other._numOfCirculations) {
		_numOfCirculations = new CuVector1<int>(other._numOfCirculations->getSize());
		*_numOfCirculations = *other._numOfCirculations;
	}

	if (other._totalCost) {
		_totalCost = new CuVector1<AmountOfMoney>(other._totalCost->getSize());
		*_totalCost = *other._totalCost;
	}

	if (other._circStartTime) {
		_circStartTime = new CuMatrix1<PointInTime>(other._circStartTime->getNumOfRows(), other._circStartTime->getNumOfCols());
		*_circStartTime = *other._circStartTime;
	}

	if (other._circCost) {
		_circCost = new CuMatrix1<AmountOfMoney>(other._circCost->getNumOfRows(), other._circCost->getNumOfCols());
		*_circCost = *other._circCost;
	}

	if (other._circCostRatio) {
		_circCostRatio = new CuMatrix1<float>(other._circCostRatio->getNumOfRows(), other._circCostRatio->getNumOfCols());
		*_circCostRatio = *other._circCostRatio;
	}
}


CuEvaluationResult::~CuEvaluationResult()
{
	if (_numOfCirculations) delete _numOfCirculations;
	if (_totalCost) delete _totalCost;
	if (_circStartTime) delete _circStartTime;
	if (_circCost) delete _circCost;
	if (_circCostRatio) delete _circCostRatio;
}


bool CuEvaluationResult::operator==(const CuEvaluationResult &rhs)
{
	if (this == &rhs) return true;

	if (_numOfSolutions != rhs._numOfSolutions) return false;
	if (*_numOfCirculations != *rhs._numOfCirculations) return false;
	if (*_totalCost != *rhs._totalCost) return false;
	if (*_circStartTime != *rhs._circStartTime) return false;
	if (*_circCost != *rhs._circCost) return false;
	if (*_circCostRatio != *rhs._circCostRatio) return false;

	return true;
}


bool CuEvaluationResult::operator!=(const CuEvaluationResult &rhs)
{
	return !(*this == rhs);
}


CU_HSTDEV void CuEvaluationResult::initialize()
{
	_numOfCirculations->initialize();
	_totalCost->initialize();
}


CU_HSTDEV void CuEvaluationResult::initialize(int solutionId)
{
	(*_numOfCirculations)[solutionId] = 0;
	setTotalCost(solutionId, AmountOfMoney(0));
}


CuEvaluationResult* CuEvaluationResult::getDevPtr()
{
	if (!_devicePtr) {
		CuEvaluationResult temp;
		temp._numOfSolutions = _numOfSolutions;
		temp._numOfCirculations= _numOfCirculations->getDevPtr();
		temp._circCost = _circCost->getDevPtr();
		temp._circCostRatio = _circCostRatio->getDevPtr();
		temp._circStartTime = _circStartTime->getDevPtr();
		temp._totalCost = _totalCost->getDevPtr();
		temp._devicePtr = 0;

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuEvaluationResult)));
		
		CUDA_CHECK(cudaMemcpy(_devicePtr, &temp, sizeof(CuEvaluationResult), cudaMemcpyHostToDevice));

		temp._numOfCirculations = 0;
		temp._circCost = 0;
		temp._circCostRatio = 0;
		temp._circStartTime = 0;
		temp._totalCost = 0;
		temp._devicePtr = 0;
	}
	return _devicePtr;
}


void CuEvaluationResult::copyToHost()
{
	if (_devicePtr) {
		_numOfCirculations->copyToHost();
		_circCost->copyToHost();
		_circCostRatio->copyToHost();
		_circStartTime->copyToHost();
		_totalCost->copyToHost();
	}
}


void CuEvaluationResult::copyToDevice()
{
	if (_devicePtr) {
		CuEvaluationResult *tempObj = _devicePtr;

		_numOfCirculations->copyToDevice();
		_circCost->copyToDevice();
		_circCostRatio->copyToDevice();
		_circStartTime->copyToDevice();
		_totalCost->copyToDevice();

		CuVector1<int> *tempNumOfCirculations = _numOfCirculations;
		CuVector1<AmountOfMoney> *tempTotalCost = _totalCost;
		CuMatrix1<PointInTime> *tempCircStartTime = _circStartTime;
		CuMatrix1<AmountOfMoney> *tempCircCost = _circCost;
		CuMatrix1<float> *tempCircCostRatio = _circCostRatio;

		_numOfCirculations = _numOfCirculations->getDevPtr();
		_circCost= _circCost->getDevPtr();
		_circCostRatio= _circCostRatio->getDevPtr();
		_circStartTime= _circStartTime->getDevPtr();
		_totalCost= _totalCost->getDevPtr();
		_devicePtr = 0;

		CUDA_CHECK(cudaMemcpy(tempObj, this, sizeof(CuEvaluationResult), cudaMemcpyHostToDevice));

		_devicePtr = tempObj;

		_numOfCirculations = tempNumOfCirculations;
		_circCost = tempCircCost;
		_circCostRatio = tempCircCostRatio;
		_circStartTime = tempCircStartTime;
		_totalCost = tempTotalCost;
	}
}


void CuEvaluationResult::dump()
{
	for (int i = 0; i < _numOfSolutions; i++) {
		cout << "Solution " << i << ":" << endl;
		cout << "numOfCirculations=" << getNumOfCirculations(i) << endl;
		cout << "totalCost=" << (int)getTotalCost(i) << endl;
		for (CirculationId c(0); c < getNumOfCirculations(i); c++) {
			cout << "  Circulation " << (short)c << ":" << endl;
			cout << "    circStartTime=" << (int)getCircStartTime(i, c) << endl;
			cout << "    circCost     =" << (int)getCircCost(i, c) << endl;
			cout << "    circCostRatio=" << getCircCostRatio(i, c) << endl;
		}
	}
}
