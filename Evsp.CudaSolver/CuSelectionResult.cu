#include "CuSelectionResult.h"
#include "cuda_runtime.h"



CuSelectionResult::CuSelectionResult(int populationSize, int maxNumOfCirculations, int maxNumOfNodes)
	: _populationSize(populationSize), _alive(0), _hasNewBestSolution(false),
	_bestSolutionKeyData(-1, AmountOfMoney(INT_MAX), INT_MAX),
	_worstSolutionKeyData(-1, AmountOfMoney(0), 0), _devicePtr(0)
{
	_alive = new CuVector1<bool>(_populationSize);
	markAllAsDead();
}


CuSelectionResult::~CuSelectionResult()
{
	if (_alive) delete _alive;
	if (_devicePtr) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


CuSelectionResult* CuSelectionResult::getDevPtr()
{
	if (!_devicePtr) {
		CuVector1<bool> *tempAlive = _alive;
		_alive = _alive->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuSelectionResult)));
		CUDA_CHECK(cudaMemcpy(_devicePtr, this, sizeof(CuSelectionResult), cudaMemcpyHostToDevice));

		_alive = tempAlive;
	}
	return _devicePtr;
}


void CuSelectionResult::copyToHost()
{
	if (_devicePtr) {
		CuVector1<bool> *tempAlive = _alive;

		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuSelectionResult), cudaMemcpyDeviceToHost));

		_alive = tempAlive;
		_alive->copyToHost();
	}
}


void CuSelectionResult::copyToDevice()
{
	if (_devicePtr) {
		_alive->copyToDevice();
		CuVector1<bool> *tempAlive = _alive;
		_alive = _alive->getDevPtr();

		CUDA_CHECK(cudaMemcpy(_devicePtr, this, sizeof(CuSelectionResult), cudaMemcpyHostToDevice));

		_alive = tempAlive;
	}
}


CU_HSTDEV bool CuSelectionResult::checkTheBestSolutionInClass(SolutionKeyData toBeChecked)
{
	_hasNewBestSolution = false;

	if (_bestSolutionKeyData.getTotalCost() > toBeChecked.getTotalCost()) {
		_bestSolutionKeyData = toBeChecked;
		_hasNewBestSolution = true;
	}

	markAsAlive(_bestSolutionKeyData.getSolutionId());

	return _hasNewBestSolution;
}


CU_HSTDEV bool CuSelectionResult::checkTheWorstSolutionInClass(SolutionKeyData toBeChecked)
{
	bool retVal = false;

	if (_worstSolutionKeyData.getTotalCost() < toBeChecked.getTotalCost()) {
		_worstSolutionKeyData = toBeChecked;
		retVal = true;
	}

	return retVal;
}


