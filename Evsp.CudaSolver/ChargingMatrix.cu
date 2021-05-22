#include "ChargingMatrix.h"
#include <iostream>
#include <stdexcept> 
#include <assert.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/Stopwatch.h"
#include "EVSP.Model/EmptyTrip.h"
#include "CudaCheck.h"
#include "RandomGpu.h"
#include "CuMatrix1.hpp"
#include "EvspLimits.h"
#include "CuProblem.h"


ChargingMatrix::ChargingMatrix(const CuProblem &problem, PlattformConfig plattform, bool performCheck)
	: _chargingMatrix(0), _devicePtr(0)
{
	switch (plattform) {
	case CPU:
		createChargingMatrixCpu(problem);
		break;
	case GPU:
		createChargingMatrixGpu(problem);
		break;
	default:
		throw runtime_error("ChargingMatrix: Unbekannte Plattform!");
	}

	if (performCheck) {
		if (!check(problem)) throw runtime_error("ChargingMatrix: Matrix ist fehlerhaft!");
	}
}


ChargingMatrix::~ChargingMatrix()
{
	if (_chargingMatrix) delete _chargingMatrix;
	if (_devicePtr) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


ChargingMatrix* ChargingMatrix::getDevPtr()
{
	if (!_devicePtr) {
		CuMatrix1<StopId> *tempMatrix = _chargingMatrix;
		ChargingMatrix* tempDevicePtr = 0;

		_chargingMatrix = _chargingMatrix->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempDevicePtr, sizeof(ChargingMatrix)));
		CUDA_CHECK(cudaMemcpy(tempDevicePtr, this, sizeof(ChargingMatrix), cudaMemcpyHostToDevice));

		_chargingMatrix = tempMatrix;
		_devicePtr = tempDevicePtr;
	}
	return _devicePtr;
}


void ChargingMatrix::copyToHost()
{
	if (_devicePtr) {
		_chargingMatrix->copyToHost();
	}
}


CU_HSTDEV StopId ChargingMatrix::getFastestToReach(StopId departure, StopId destination) const
{
	return _chargingMatrix->get((short)departure, (short)destination);
}


__global__ void createChargingMatrixKernel(CuMatrix1<StopId> *matrix, int numOfChargingStations, CuStops *stops, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx >= matrix->getNumOfRows()) return;
	if (ty >= matrix->getNumOfCols()) return;

	StopId fromId = StopId(tx);
	StopId toId = StopId(ty);
	StopId chargingStation = StopId::invalid();

	EmptyTripId trip1_id, trip2_id;
	DurationInSeconds nearestRuntime = DurationInSeconds(INT_MAX);
	DurationInSeconds runtime;

	for (ChargingStationId i = ChargingStationId(0); i < ChargingStationId(numOfChargingStations); i++) {
		StopId id = stops->getStopIdOfChargingStation(i);
		CuEmptyTrip trip1, trip2;
		if (connectionMatrix->getEmptyTrip(fromId, id, trip1) && connectionMatrix->getEmptyTrip(id, toId, trip2)) {
			runtime = trip1.duration + trip2.duration;
			if (runtime < nearestRuntime) {
				chargingStation = id;
				nearestRuntime = runtime;
			}
		}
	}

	matrix->set((short)fromId, (short)toId, chargingStation);
}


void ChargingMatrix::createChargingMatrixCpu(const CuProblem &problem)
{
	Stopwatch stopwatch;
	stopwatch.start();
	
	const ConnectionMatrix &connectionMatrix = problem.getConnectionMatrix();
	const CuStops &stops = problem.getStops();

	int numOfStops = problem.getStops().getNumOfStops();
	int numOfChargingStations = stops.getNumOfChargingStations();
	_chargingMatrix = new CuMatrix1<StopId>(numOfStops, numOfStops);

	for (StopId from(0); from < numOfStops; from++) {
		for (StopId to(0); to < numOfStops; to++) {

			StopId chargingStation = StopId::invalid();

			EmptyTripId trip1_id, trip2_id;
			DurationInSeconds nearestRuntime = DurationInSeconds(INT_MAX);
			DurationInSeconds runtime;

			for (ChargingStationId i = ChargingStationId(0); i < ChargingStationId(numOfChargingStations); i++) {
				StopId id = stops.getStopIdOfChargingStation(i);
				shared_ptr<CuEmptyTrip> trip1 = connectionMatrix.getEmptyTrip(from, id);
				shared_ptr<CuEmptyTrip> trip2 = connectionMatrix.getEmptyTrip(id, to);
				if (trip1 && trip2) {
					runtime = trip1->duration + trip2->duration;
					if (runtime < nearestRuntime) {
						chargingStation = id;
						nearestRuntime = runtime;
					}
				}
			}

			_chargingMatrix->set((short)from, (short)to, chargingStation);
		}
	}

	stopwatch.stop("ChargingMatrix erzeugt (CPU): ");
}


void ChargingMatrix::createChargingMatrixGpu(const CuProblem &problem)
{
	Stopwatch stopwatch;
	stopwatch.start();

	int numOfStops = problem.getStops().getNumOfStops();
	_chargingMatrix = new CuMatrix1<StopId>(numOfStops, numOfStops);
	// Die Matrix muss nicht initialisiert werden, da der Kernel alle Zellen belegt.

	int numOfChargingStations = problem.getStops().getNumOfChargingStations();
	CuStops &stops = problem.getStops();
	CuEmptyTrips &emptyTrips = problem.getEmptyTrips();
	ConnectionMatrix &connectionMatrix = problem.getConnectionMatrix();

	const int blockSize = 32; // Threads pro Block
	int numOfBlocks = (numOfStops + blockSize - 1) / blockSize;
	dim3 dimGrid(numOfBlocks, numOfBlocks);
	dim3 dimBlock(blockSize, blockSize);
	createChargingMatrixKernel << <dimGrid, dimBlock >> > (_chargingMatrix->getDevPtr(), numOfChargingStations, stops.getDevPtr(), emptyTrips.getDevPtr(), connectionMatrix.getDevPtr());
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	_chargingMatrix->copyToHost();

	stopwatch.stop("ChargingMatrix erzeugt (GPU): ");
}


bool ChargingMatrix::check(const CuProblem &problem)
{
	Stopwatch stopwatch;
	stopwatch.start();

	bool retVal = true;

	int numOfStops = problem.getStops().getNumOfStops();

	for (StopId from(0); from < numOfStops; from++) {
		for (StopId to(0); to < numOfStops; to++) {

			StopId csId = getFastestToReach(from, to);

			if (csId.isValid()) {
				const CuStop &cs = problem.getStops().getStop(csId);
				if (!cs.isChargingStation) {
					cerr << "ChargingMatrix::check: Die Haltestelle " << (short)csId << " ist keine Ladestation" << endl;
					retVal = false;
				}

				if (!problem.getConnectionMatrix().connected(from, csId)) {
					cerr << "ChargingMatrix::check: Die Ladestation " << (short)csId << " ist von der Haltestelle "
						<< (short)from << " aus nicht zu erreichen!" << endl;
					retVal = false;
				}

				if (!problem.getConnectionMatrix().connected(csId, to)) {
					cerr << "ChargingMatrix::check: Die Haltestelle " << (short)to << " ist von der Ladestation "
						<< (short)csId << " aus nicht zu erreichen!" << endl;
					retVal = false;
				}
			}
		}
	}

	stopwatch.stop("ChargingMatrix geprüft: ");

	return retVal;
}
