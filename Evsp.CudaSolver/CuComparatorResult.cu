#include "CuComparatorResult.h"
#include "cuda_runtime.h"
#include "CuSolutionComparator.h"




bool CirculationKeyData::operator==(const CirculationKeyData &rhs)
{
	if (this == &rhs) return true;

	if (solutionId != rhs.solutionId) return false;
	if (circulationId != rhs.circulationId) return false;
	if (value != rhs.value) return false;

	return true;
}


std::string CirculationKeyData::toString()
{
	return "solutionId=" + to_string(solutionId) + ", circulationId=" + to_string((short)circulationId) + ", value=" + to_string(value);
}


bool CirculationKeyData::operator!=(const CirculationKeyData &rhs)
{
	return !(*this == rhs);
}


CuComparatorResult::CuComparatorResult(const CuComparatorResult &other)
	: _devicePtr(0)
{
	_lowestTotalCost = other._lowestTotalCost;
	_highestTotalCost = other._highestTotalCost;
	_lowestNumOfVehicles = other._lowestNumOfVehicles;
	_highestNumOfVehicles = other._highestNumOfVehicles;
	_lowestCircCostRatio = other._lowestCircCostRatio;
	_highestCircCostRatio = other._highestCircCostRatio;
}


CuComparatorResult::~CuComparatorResult()
{
	if (_devicePtr) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


CuComparatorResult& CuComparatorResult::operator=(const CuComparatorResult &rhs)
{
	if (this != &rhs) {
		_lowestTotalCost = rhs._lowestTotalCost;
		_highestTotalCost = rhs._highestTotalCost;
		_lowestNumOfVehicles = rhs._lowestNumOfVehicles;
		_highestNumOfVehicles = rhs._highestNumOfVehicles;
		_lowestCircCostRatio = rhs._lowestCircCostRatio;
		_highestCircCostRatio = rhs._highestCircCostRatio;
	}
	return *this;
}


bool CuComparatorResult::operator==(const CuComparatorResult &rhs)
{
	if (this == &rhs) return true;

	if (_lowestTotalCost != rhs._lowestTotalCost) return false;
	if (_highestTotalCost != rhs._highestTotalCost) return false;
	if (_lowestNumOfVehicles != rhs._lowestNumOfVehicles) return false;
	if (_highestNumOfVehicles != rhs._highestNumOfVehicles) return false;
	if (_lowestCircCostRatio != rhs._lowestCircCostRatio) return false;
	if (_highestCircCostRatio != rhs._highestCircCostRatio) return false;

	return true;
}


bool CuComparatorResult::operator!=(const CuComparatorResult &rhs)
{
	return !(*this == rhs);
}


CuComparatorResult* CuComparatorResult::getDevPtr()
{
	if (!_devicePtr) {
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuComparatorResult)));
		CUDA_CHECK(cudaMemcpy(_devicePtr, this, sizeof(CuComparatorResult), cudaMemcpyHostToDevice));
	}
	return _devicePtr;
}


void CuComparatorResult::copyToHost()
{
	if (_devicePtr) {
		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuComparatorResult), cudaMemcpyDeviceToHost));
	}
}


void CuComparatorResult::copyToDevice()
{
	if (_devicePtr) {
		CUDA_CHECK(cudaMemcpy(_devicePtr, this, sizeof(CuComparatorResult), cudaMemcpyHostToDevice));
	}
}


void CuComparatorResult::initialize()
{
	_lowestTotalCost = SolutionKeyData();
	_highestTotalCost = SolutionKeyData();
	_lowestNumOfVehicles = SolutionKeyData();
	_highestNumOfVehicles = SolutionKeyData();
	_lowestCircCostRatio = CirculationKeyData();
	_highestCircCostRatio = CirculationKeyData();
}


void CuComparatorResult::dump()
{
	cout << _lowestTotalCost.toString() << endl;
	//cout << "lowestTotalCost:      " << _lowestTotalCost.toString() << endl;
	//cout << "highestTotalCost:     " << _highestTotalCost.toString() << endl;
	//cout << "lowestNumOfVehicles:  " << _lowestNumOfVehicles.toString() << endl;
	//cout << "highestNumOfVehicles: " << _highestNumOfVehicles.toString() << endl;
	//cout << "lowestCircCostRatio:  " << _lowestCircCostRatio.toString() << endl;
	//cout << "highestCircCostRatio: " << _highestCircCostRatio.toString() << endl;
}
