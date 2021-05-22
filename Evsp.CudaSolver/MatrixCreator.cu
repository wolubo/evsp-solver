#include "MatrixCreator.h"

#include <iostream>
#include <stdexcept> 
#include <assert.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/Stopwatch.h"
#include "EVSP.Model/EmptyTrip.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CudaCheck.h"
#include "RandomGpu.h"
#include "CuMatrix1.hpp"
#include "EvspLimits.h"


__global__ void createRandomMatrixKernel(int numOfRows, int numOfCols, RandomMatrix* rndMatrix, RandomGpu *random)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < numOfCols) {

		for (ServiceTripId i = ServiceTripId(0); i < ServiceTripId(numOfRows); i++) {
			rndMatrix->set((short)i, col, i);
		}

		// Fisher-Yates-Shuffle
		for (int i = numOfRows - 1; i > 0; i--) {
			int index = random->rand(i, col);
			ServiceTripId a = rndMatrix->get(index, col);
			ServiceTripId b = rndMatrix->get(i, col);
			rndMatrix->set(i, col, a);
			rndMatrix->set(index, col, b);
		}
	}
}


shared_ptr<RandomMatrix> MatrixCreator::createRandomMatrix(int numOfServiceTrips, int populationSize, shared_ptr<RandomGpu> rand)
{
	Stopwatch stopwatch;
	stopwatch.start();

	assert(rand);
	assert(rand->getSize() >= populationSize);

	// Es wird eine CuMatrix1 auf dem Device benötigt, die mit -1 initialisiert ist, 
	// damit Kollisionen beim Schreiben der Zufallswerte erkannt werden können.
	shared_ptr<RandomMatrix> matrix = make_shared<RandomMatrix>(numOfServiceTrips, populationSize);
	matrix->setAllOnDevice(ServiceTripId::invalid());

	int blockSize; // Threads pro Block
	int minGridSize;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)createRandomMatrixKernel));
	int numOfBlocks = (numOfServiceTrips + blockSize - 1) / blockSize;
	dim3 dimGrid(numOfBlocks);
	dim3 dimBlock(blockSize);
	createRandomMatrixKernel << <dimGrid, dimBlock >> > (numOfServiceTrips, populationSize, matrix->getDevPtr(), rand->getDevPtr());
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	matrix->copyToHost();

	stopwatch.stop("RandomMatrix erzeugt: ");
	return matrix;
}


#ifdef _DEBUG
bool MatrixCreator::checkRandomMatrix(int numOfRows, shared_ptr<RandomMatrix> rndMatrix)
{
	Stopwatch stopwatch;
	stopwatch.start();

	rndMatrix->copyToHost();

	bool retVal = true;

	bool *ba = new bool[rndMatrix->getNumOfRows()];
	for (int col = 0; col < rndMatrix->getNumOfCols(); col++) {
		for (int i = 0; i < numOfRows; i++) ba[i] = false;
		for (int row = 0; row < numOfRows; row++) {
			ServiceTripId value = rndMatrix->get(row, col);
			if (ba[(short)value]) {
				std::cerr << "checkRandomMatrix: Doppelter Wert! Col=" << col << " Row=" << row << " Value=" << (short)value << std::endl;
				retVal = false;
			}
			ba[(short)value] = true;
		}
	}
	delete[] ba;

	stopwatch.stop("RandomMatrix geprüft: ");

	return retVal;
}
#endif



shared_ptr<VehicleTypeGroupIntersection> MatrixCreator::createVehicleTypeGroupIntersection(std::shared_ptr<CuProblem> problem)
{
	shared_ptr<VehicleTypeGroupIntersection> retVal = make_shared<VehicleTypeGroupIntersection>();
	retVal->setAll(false);
	ushort numOfVTG = problem->getVehicleTypeGroups().getNumOfVehicleTypeGroups();
	// TODO Optimieren: CuMatrix2 ist spiegelsymmetrisch.
	for (VehicleTypeGroupId row = VehicleTypeGroupId(0); row < VehicleTypeGroupId(numOfVTG); row++) {
		for (VehicleTypeGroupId col = VehicleTypeGroupId(0); col < VehicleTypeGroupId(numOfVTG); col++) {
			bool hit = false;
			if (row != col) {
				ushort i = 0;
				while (i < problem->getVehicleTypeGroups().getNumOfVehicleTypes(row) && !hit) {
					VehicleTypeId vt_id = problem->getVehicleTypeGroups().get(row, i);
					hit = problem->getVehicleTypeGroups().hasVehicleType(col, vt_id);
					i++;
				}
			}
			else {
				hit = true;
			}
			if (hit) {
				retVal->set((short)row, (short)col, true);
			}
		}
	}
	return retVal;
}

