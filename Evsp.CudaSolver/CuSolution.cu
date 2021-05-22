#include "CuSolution.h"

#include "CudaCheck.h"

#include <cuda_runtime.h>
#include <cuda.h> 








CuSolution* CuSolution::createOnDevice(int initialCapacity, int growRate)
{
	CuSolution *newItem_hst = new CuSolution();
	newItem_hst->_capacity = initialCapacity;
	newItem_hst->_growRate = growRate;
	newItem_hst->_numOfCirculations = 0;

	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(newItem_hst->_circulations), sizeof(Solution*) * initialCapacity));
	

	CuSolution *newItem_dev;
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(newItem_dev), sizeof(CuSolution)));
	
	CUDA_CHECK(cudaMemcpy(newItem_dev, newItem_hst, sizeof(CuSolution), cudaMemcpyHostToDevice));

	delete newItem_hst;

	return newItem_dev;
}


void CuSolution::deleteOnDevice(CuSolution** devicePtr)
{
	if (devicePtr != 0 && *devicePtr != 0) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, *devicePtr));
		*devicePtr = 0;
	}
}


CuSolution* CuSolution::copy2host(CuSolution* devicePtr)
{
	// CuSolution-Objekt auf den Host kopieren.
	CuSolution *newItem_hst = new CuSolution();
	CUDA_CHECK(cudaMemcpy(newItem_hst, devicePtr, sizeof(CuSolution), cudaMemcpyDeviceToHost));

	// Das Array mit den Solution-Pointern auf den Host kopieren.
	Solution** circulations_dev = newItem_hst->_circulations;
	newItem_hst->_circulations = new Solution*[newItem_hst->_capacity];
	CUDA_CHECK(cudaMemcpy(newItem_hst->_circulations, circulations_dev, sizeof(Solution*) * newItem_hst->_capacity, cudaMemcpyDeviceToHost));

	// Inhalt des Arrays auf den Host kopieren.
	for (int i = 0; i < newItem_hst->_numOfCirculations; i++) {
//		newItem_hst->_circulations[i]->copy2host();
	}

	return newItem_hst;
}


Solution* CuSolution::getCirculation(int index)
{
	if (index >= 0 && index < _numOfCirculations) return _circulations[index];
	return false;
}


void CuSolution::addCirculation(Solution* newCirculation)
{
	if (newCirculation == 0) return;

	if (_numOfCirculations == _capacity) {
		// Die Kapazitätsgrenze ist erreicht. Vor dem Hinzufügen muss zusätzlicher Platz geschaffen werden.
		int newCapacity = _capacity + _growRate;
		Solution **newArray;
		//newArray = new Solution*[newCapacity];
		wbCudaMalloc(__FILE__, __LINE__, (void**)&(newArray), sizeof(Solution*) * newCapacity);
		for (int i = 0; i < _numOfCirculations; i++) newArray[i] = _circulations[i];
		//delete[] _circulations;
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _circulations));
		_circulations = newArray;
		_capacity = newCapacity;
	}

	_circulations[_numOfCirculations++] = newCirculation;
}


void CuSolution::removeCirculation(int index)
{
	if (index >= 0 && index < _numOfCirculations) {
		_circulations[index] = _circulations[--_numOfCirculations];
	}
}


