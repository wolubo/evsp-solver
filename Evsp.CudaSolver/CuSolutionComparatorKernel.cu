#include "CuSolutionComparatorKernel.h"
#include "cuda_runtime.h"
#include "CuComparatorResult.h"

// Blockgrösse. Muss eine Potenz von 2 sein!
#define SOLUTION_COMPARATOR_BLOCKSIZE 512


// Datenstruktur für das Speichern von Zwischenergebnissen.
struct IntermediateResults {
	IntermediateResults(int theNumOfBlocks);
	~IntermediateResults();
	IntermediateResults* getDevPtr();

	SolutionKeyData IntermediateResults::getLowestTotalCost();
	SolutionKeyData IntermediateResults::getHighestTotalCost();
	SolutionKeyData IntermediateResults::getLowestNumOfVehicles();
	SolutionKeyData IntermediateResults::getHighestNumOfVehicles();
	CirculationKeyData IntermediateResults::getLowestCircCostRatio();
	CirculationKeyData IntermediateResults::getHighestCircCostRatio();

	int	numOfBlocks;

	SolutionKeyData *lowestTotalCost;
	SolutionKeyData *highestTotalCost;
	SolutionKeyData *lowestNumOfVehicles;
	SolutionKeyData *highestNumOfVehicles;
	CirculationKeyData *lowestCircCostRatio;
	CirculationKeyData *highestCircCostRatio;

	IntermediateResults *_devPtr;
};


__global__ void solutionComparatorKernel(int populationSize, CuEvaluationResult *results, IntermediateResults *ir);

void startSolutionComparatorKernel(int populationSize, CuEvaluationResult *evaluationResults, CuComparatorResult *results)
{
	assert(evaluationResults);
	assert(results);

	int numOfBlocks = (populationSize + SOLUTION_COMPARATOR_BLOCKSIZE - 1) / SOLUTION_COMPARATOR_BLOCKSIZE;

	dim3 dimGrid(numOfBlocks, 1);
	dim3 dimBlock(SOLUTION_COMPARATOR_BLOCKSIZE, 1);

	IntermediateResults iResults(numOfBlocks);
	IntermediateResults *iResults_dev = iResults.getDevPtr();
	assert(iResults_dev);

	// Stelle die Daten zur besten und zur schlechtesten Lösung zusammen und liefere sie in iResults zurück.
	solutionComparatorKernel << <dimGrid, dimBlock >> > (populationSize, evaluationResults->getDevPtr(), iResults_dev);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	results->_lowestTotalCost = iResults.getLowestTotalCost();
	results->_highestTotalCost = iResults.getHighestTotalCost();
	results->_lowestNumOfVehicles = iResults.getLowestNumOfVehicles();
	results->_highestNumOfVehicles = iResults.getHighestNumOfVehicles();
	results->_lowestCircCostRatio = iResults.getLowestCircCostRatio();
	results->_lowestCircCostRatio = iResults.getLowestCircCostRatio();
	results->_highestCircCostRatio = iResults.getHighestCircCostRatio();
	results->_highestCircCostRatio = iResults.getHighestCircCostRatio();
}


IntermediateResults::IntermediateResults(int theNumOfBlocks)
	: _devPtr(0)
{
	numOfBlocks = theNumOfBlocks;
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&lowestTotalCost, sizeof(SolutionKeyData)*numOfBlocks));
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&highestTotalCost, sizeof(SolutionKeyData)*numOfBlocks));
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&lowestNumOfVehicles, sizeof(SolutionKeyData)*numOfBlocks));
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&highestNumOfVehicles, sizeof(SolutionKeyData)*numOfBlocks));
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&lowestCircCostRatio, sizeof(CirculationKeyData)*numOfBlocks));
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&highestCircCostRatio, sizeof(CirculationKeyData)*numOfBlocks));
}


IntermediateResults::~IntermediateResults()
{
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, lowestTotalCost));
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, highestTotalCost));
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, lowestNumOfVehicles));
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, highestNumOfVehicles));
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, lowestCircCostRatio));
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, highestCircCostRatio));
}


IntermediateResults* IntermediateResults::getDevPtr()
{
	if (!_devPtr) {
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devPtr, sizeof(IntermediateResults)));
		CUDA_CHECK(cudaMemcpy(_devPtr, this, sizeof(IntermediateResults), cudaMemcpyHostToDevice));
	}
	return _devPtr;
}


SolutionKeyData IntermediateResults::getLowestTotalCost()
{
	SolutionKeyData temp;
	CUDA_CHECK(cudaMemcpy(&temp, lowestTotalCost, sizeof(SolutionKeyData), cudaMemcpyDeviceToHost));
	return temp;
}

SolutionKeyData IntermediateResults::getHighestTotalCost()
{
	SolutionKeyData temp;
	CUDA_CHECK(cudaMemcpy(&temp, highestTotalCost, sizeof(SolutionKeyData), cudaMemcpyDeviceToHost));
	return temp;
}


SolutionKeyData IntermediateResults::getLowestNumOfVehicles()
{
	SolutionKeyData temp;
	CUDA_CHECK(cudaMemcpy(&temp, lowestNumOfVehicles, sizeof(SolutionKeyData), cudaMemcpyDeviceToHost));
	return temp;
}


SolutionKeyData IntermediateResults::getHighestNumOfVehicles()
{
	SolutionKeyData temp;
	CUDA_CHECK(cudaMemcpy(&temp, highestNumOfVehicles, sizeof(SolutionKeyData), cudaMemcpyDeviceToHost));
	return temp;
}


CirculationKeyData IntermediateResults::getLowestCircCostRatio()
{
	CirculationKeyData temp;
	CUDA_CHECK(cudaMemcpy(&temp, lowestCircCostRatio, sizeof(CirculationKeyData), cudaMemcpyDeviceToHost));
	return temp;
}


CirculationKeyData IntermediateResults::getHighestCircCostRatio()
{
	CirculationKeyData temp;
	CUDA_CHECK(cudaMemcpy(&temp, highestCircCostRatio, sizeof(CirculationKeyData), cudaMemcpyDeviceToHost));
	return temp;
}


__global__ void solutionComparatorKernel(int populationSize, CuEvaluationResult *results, IntermediateResults *ir)
{
	__shared__ SolutionKeyData lowestTotalCost[SOLUTION_COMPARATOR_BLOCKSIZE];
	__shared__ SolutionKeyData highestTotalCost[SOLUTION_COMPARATOR_BLOCKSIZE];
	__shared__ SolutionKeyData lowestNumOfVehicles[SOLUTION_COMPARATOR_BLOCKSIZE];
	__shared__ SolutionKeyData highestNumOfVehicles[SOLUTION_COMPARATOR_BLOCKSIZE];
	__shared__ CirculationKeyData lowestCircCostRatio[SOLUTION_COMPARATOR_BLOCKSIZE];
	__shared__ CirculationKeyData highestCircCostRatio[SOLUTION_COMPARATOR_BLOCKSIZE];

	__syncthreads(); // Einige Member haben nicht-trivialen Konstruktoren --> Race-conditions!

	int tx = threadIdx.x;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Speichere die Ids und die Kosten der Lösungen des betrachteten Bereichs im Shared memory. Daran nehmen alle Threads teil.
	if (tid < populationSize) {
		AmountOfMoney totalCost = results->getTotalCost(tid);
		assert(totalCost > 0);

		int numOfVehicles = results->getNumOfCirculations(tid);
		assert(numOfVehicles > 0);

		lowestTotalCost[tx] = SolutionKeyData(tid, totalCost, numOfVehicles);
		highestTotalCost[tx] = SolutionKeyData(tid, totalCost, numOfVehicles);

		lowestNumOfVehicles[tx] = SolutionKeyData(tid, totalCost, numOfVehicles);
		highestNumOfVehicles[tx] = SolutionKeyData(tid, totalCost, numOfVehicles);

		float temp;
		for (CirculationId circId(0); circId < results->getNumOfCirculations(tid); circId++) {
			temp = results->getCircCostRatio(tid, circId);
			assert(temp >= 1.0f);
			if (temp < lowestCircCostRatio[tx].value) {
				lowestCircCostRatio[tx] = CirculationKeyData(tid, circId, temp);
			}
			if (temp > highestCircCostRatio[tx].value) {
				highestCircCostRatio[tx] = CirculationKeyData(tid, circId, temp);
			}
		}
	}
	else {
		// Initialisiere die restlichen Bereiche des Shared memory.
		lowestTotalCost[tx] = SolutionKeyData(-1, AmountOfMoney(INT_MAX), INT_MAX);
		highestTotalCost[tx] = SolutionKeyData(-1, AmountOfMoney(0), 0);
		lowestNumOfVehicles[tx] = SolutionKeyData(-1, AmountOfMoney(INT_MAX), INT_MAX);
		highestNumOfVehicles[tx] = SolutionKeyData(-1, AmountOfMoney(0), 0);
		lowestCircCostRatio[tx] = CirculationKeyData(-1, CirculationId::invalid(), FLT_MAX);
		highestCircCostRatio[tx] = CirculationKeyData(-1, CirculationId::invalid(), 0.0f);
	}

	__syncthreads();

	// Suche die Lösung mit den geringsten Gesamtkosten im betrachteten Bereich. Nur die Hälfte aller Threads nimmt anfangs teil.
	// Die Anzahl der teilnehmenden Threads halbiert sich in jedem Durchlauf.
	//int stride_prev = 0;
	for (int stride = 1; stride <= SOLUTION_COMPARATOR_BLOCKSIZE / 2; stride <<= 1) {
		if (tx % (stride * 2) == 0) {
			if ((tx + stride) < SOLUTION_COMPARATOR_BLOCKSIZE) {

				if (lowestTotalCost[tx].getTotalCost() > lowestTotalCost[tx + stride].getTotalCost()) {
					lowestTotalCost[tx] = lowestTotalCost[tx + stride];
				}

				if (highestTotalCost[tx].getTotalCost() < highestTotalCost[tx + stride].getTotalCost()) {
					highestTotalCost[tx] = highestTotalCost[tx + stride];
				}

				if (lowestNumOfVehicles[tx].getNumOfCirculations() > lowestNumOfVehicles[tx + stride].getNumOfCirculations()) {
					lowestNumOfVehicles[tx] = lowestNumOfVehicles[tx + stride];
				}
				else if(lowestNumOfVehicles[tx].getNumOfCirculations() == lowestNumOfVehicles[tx + stride].getNumOfCirculations()) {
					if (lowestNumOfVehicles[tx].getTotalCost() > lowestNumOfVehicles[tx + stride].getTotalCost()) {
						lowestNumOfVehicles[tx] = lowestNumOfVehicles[tx + stride];
					}
				}

				if (highestNumOfVehicles[tx].getNumOfCirculations() < highestNumOfVehicles[tx + stride].getNumOfCirculations()) {
					highestNumOfVehicles[tx] = highestNumOfVehicles[tx + stride];
				}
				else if (highestNumOfVehicles[tx].getNumOfCirculations() == highestNumOfVehicles[tx + stride].getNumOfCirculations()) {
					if (highestNumOfVehicles[tx].getTotalCost() < highestNumOfVehicles[tx + stride].getTotalCost()) {
						highestNumOfVehicles[tx] = highestNumOfVehicles[tx + stride];
					}
				}

				if (lowestCircCostRatio[tx].value > lowestCircCostRatio[tx + stride].value) {
					lowestCircCostRatio[tx] = lowestCircCostRatio[tx + stride];
				}

				if (highestCircCostRatio[tx].value < highestCircCostRatio[tx + stride].value) {
					highestCircCostRatio[tx] = highestCircCostRatio[tx + stride];
				}

			}
		}
		__syncthreads();
	}

	// Kopiere die beste Lösung dieses Blocks in den Zwischenlösungs-Puffer.
	if (tx == 0) {
		ir->lowestTotalCost[blockIdx.x] = lowestTotalCost[0];
		ir->highestTotalCost[blockIdx.x] = highestTotalCost[0];
		ir->lowestNumOfVehicles[blockIdx.x] = lowestNumOfVehicles[0];
		ir->highestNumOfVehicles[blockIdx.x] = highestNumOfVehicles[0];
		ir->lowestCircCostRatio[blockIdx.x] = lowestCircCostRatio[0];
		ir->highestCircCostRatio[blockIdx.x] = highestCircCostRatio[0];
	}

	__syncthreads();

	// Suche im Zwischenlösungs-Puffer nach dem besten Ergebnis aller Blöcke und lege es unter dem Index 0 im Zwischenlösungs-Puffer ab.
	assert(ir->numOfBlocks < blockDim.x);
	if (blockIdx.x == 0) {
		if (tx < ir->numOfBlocks) {
			lowestTotalCost[tx] = ir->lowestTotalCost[tx];
			highestTotalCost[tx] = ir->highestTotalCost[tx];
			lowestNumOfVehicles[tx] = ir->lowestNumOfVehicles[tx];
			highestNumOfVehicles[tx] = ir->highestNumOfVehicles[tx];
			lowestCircCostRatio[tx] = ir->lowestCircCostRatio[tx];
			highestCircCostRatio[tx] = ir->highestCircCostRatio[tx];
		}
		else {
			// Initialisiere die restlichen Bereiche des Shared memory.
			lowestTotalCost[tx] = SolutionKeyData(-1, AmountOfMoney(INT_MAX), INT_MAX);
			highestTotalCost[tx] = SolutionKeyData(-1, AmountOfMoney(0), 0);
			lowestNumOfVehicles[tx] = SolutionKeyData(-1, AmountOfMoney(INT_MAX), INT_MAX);
			highestNumOfVehicles[tx] = SolutionKeyData(-1, AmountOfMoney(0), 0);
			lowestCircCostRatio[tx] = CirculationKeyData(-1, CirculationId::invalid(), FLT_MAX);
			highestCircCostRatio[tx] = CirculationKeyData(-1, CirculationId::invalid(), 0.0f);
		}
	}

	__syncthreads();

	// Hier nehmen zwar alle Threads teil, es interessiert aber nur der Inhalt des shared memory des Blocks 0.
	for (int stride = 1; stride <= SOLUTION_COMPARATOR_BLOCKSIZE / 2; stride <<= 1) {
		if (tx % (stride * 2) == 0) {
			if ((tx + stride) < SOLUTION_COMPARATOR_BLOCKSIZE) {

				if (lowestTotalCost[tx].getTotalCost() > lowestTotalCost[tx + stride].getTotalCost()) {
					lowestTotalCost[tx] = lowestTotalCost[tx + stride];
				}

				if (highestTotalCost[tx].getTotalCost() < highestTotalCost[tx + stride].getTotalCost()) {
					highestTotalCost[tx] = highestTotalCost[tx + stride];
				}

				if (lowestNumOfVehicles[tx].getNumOfCirculations() > lowestNumOfVehicles[tx + stride].getNumOfCirculations()) {
					lowestNumOfVehicles[tx] = lowestNumOfVehicles[tx + stride];
				}
				else if (lowestNumOfVehicles[tx].getNumOfCirculations() == lowestNumOfVehicles[tx + stride].getNumOfCirculations()) {
					if (lowestNumOfVehicles[tx].getTotalCost() > lowestNumOfVehicles[tx + stride].getTotalCost()) {
						lowestNumOfVehicles[tx] = lowestNumOfVehicles[tx + stride];
					}
				}

				if (highestNumOfVehicles[tx].getNumOfCirculations() < highestNumOfVehicles[tx + stride].getNumOfCirculations()) {
					highestNumOfVehicles[tx] = highestNumOfVehicles[tx + stride];
				}
				else if (highestNumOfVehicles[tx].getNumOfCirculations() == highestNumOfVehicles[tx + stride].getNumOfCirculations()) {
					if (highestNumOfVehicles[tx].getTotalCost() < highestNumOfVehicles[tx + stride].getTotalCost()) {
						highestNumOfVehicles[tx] = highestNumOfVehicles[tx + stride];
					}
				}

				if (lowestCircCostRatio[tx].value > lowestCircCostRatio[tx + stride].value) {
					lowestCircCostRatio[tx] = lowestCircCostRatio[tx + stride];
				}

				if (highestCircCostRatio[tx].value < highestCircCostRatio[tx + stride].value) {
					highestCircCostRatio[tx] = highestCircCostRatio[tx + stride];
				}

			}
		}
		__syncthreads();
	}

	if (blockIdx.x == 0) {
		ir->lowestTotalCost[0] = lowestTotalCost[0];
		ir->highestTotalCost[0] = highestTotalCost[0];
		ir->lowestNumOfVehicles[0] = lowestNumOfVehicles[0];
		ir->highestNumOfVehicles[0] = highestNumOfVehicles[0];
		ir->lowestCircCostRatio[0] = lowestCircCostRatio[0];
		ir->highestCircCostRatio[0] = highestCircCostRatio[0];
	}
}
