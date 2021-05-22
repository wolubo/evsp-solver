#include "PrevNextMatrix.h"
#include "CuProblem.h"
#include <iostream>
#include <stdexcept> 
#include <assert.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "EVSP.Model/EmptyTrip.h"
#include "EVSP.BaseClasses/Stopwatch.h"
#include "CudaCheck.h"
#include "RandomGpu.h"
#include "EvspLimits.h"


PrevNextMatrix::PrevNextMatrix(CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix, PlattformConfig plattform, bool performChecks)
	: _devicePtr(0)
{
	int numOfServiceTrips = serviceTrips->getNumOfServiceTrips();
	_pnMatrix = new CuMatrix1<DurationInSeconds>(numOfServiceTrips, numOfServiceTrips);
	_pnMatrix->setAll(DurationInSeconds::invalid());

	switch (plattform) {
	case PlattformConfig::CPU:
		createCpu(serviceTrips, emptyTrips, connectionMatrix, numOfServiceTrips);
		break;
	case PlattformConfig::GPU:
		createGpu(serviceTrips, emptyTrips, connectionMatrix, numOfServiceTrips);
		break;
	default:
		assert(false);
	}

	if (performChecks) {
		check(serviceTrips, emptyTrips, connectionMatrix);
	}
}


PrevNextMatrix::~PrevNextMatrix()
{
	if (_pnMatrix) delete _pnMatrix;
	if (_devicePtr)
	{
#ifdef __CUDACC__
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
#endif
	}
}


PrevNextMatrix* PrevNextMatrix::getDevPtr()
{
	if (!_devicePtr) {
		CuMatrix1<DurationInSeconds> *tempMatrix = _pnMatrix;
		PrevNextMatrix* tempDevicePtr = 0;

		_pnMatrix = _pnMatrix->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempDevicePtr, sizeof(ConnectionMatrix)));
		CUDA_CHECK(cudaMemcpy(tempDevicePtr, this, sizeof(ConnectionMatrix), cudaMemcpyHostToDevice));

		_pnMatrix = tempMatrix;
		_devicePtr = tempDevicePtr;
	}
	return _devicePtr;
}


void PrevNextMatrix::copyToHost()
{
	if (_devicePtr) {
		_pnMatrix->copyToHost();
	}
}


CU_HSTDEV bool PrevNextMatrix::checkCombination(ServiceTripId tripA, ServiceTripId tripB) const
{
	return _pnMatrix->get((short)tripA, (short)tripB).isValid();
}


CU_HSTDEV DurationInSeconds PrevNextMatrix::getInterval(ServiceTripId tripA, ServiceTripId tripB) const
{
	return _pnMatrix->get((short)tripA, (short)tripB);
}


void PrevNextMatrix::createCpu(CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix, int numOfServiceTrips)
{
	Stopwatch stopwatch;
	stopwatch.start();

	for (ServiceTripId prevId(0); prevId < numOfServiceTrips; prevId++) {
		for (ServiceTripId nextId(0); nextId < numOfServiceTrips; nextId++) {
			DurationInSeconds &pnItem = _pnMatrix->itemAt((short)prevId, (short)nextId);

			if (prevId == nextId) {
				// Eine Servicefahrt kann natürlich nicht vor oder nach sich selbst durchgeführt werden.
				pnItem = DurationInSeconds::invalid();
				continue;
			}

			const CuServiceTrip &prevServTrip = serviceTrips->getServiceTrip(prevId);
			const CuServiceTrip &nextServTrip = serviceTrips->getServiceTrip(nextId);

			if (prevServTrip.arrival >= nextServTrip.departure)
			{
				// Fahrt A endet erst, nachdem Fahrt B begonnen hat.
				pnItem = DurationInSeconds::invalid();
				continue;
			}

			DurationInSeconds emptyTripDuration(0);
			if (prevServTrip.toStopId != nextServTrip.fromStopId) {
				// Es ist eine Verbindungsfahrt nötig, um von der Endhaltestelle der Fahrt A zu Starthaltestelle der Fahrt B zu kommen.
				EmptyTripId et_id = connectionMatrix->getEmptyTripId(prevServTrip.toStopId, nextServTrip.fromStopId);
				if (!et_id.isValid()) {
					// Die nötige Verbingungsfahrt existiert nicht.
					pnItem = DurationInSeconds::invalid();
					continue;
				}
				else {
					emptyTripDuration = emptyTrips->getEmptyTrip(et_id).duration;
				}
			}

			DurationInSeconds intervall = DurationInSeconds((int)nextServTrip.departure - (int)prevServTrip.arrival - 2);

			if (intervall < emptyTripDuration) {
				// Der Zeitraum zwischen den beiden Servicefahrten reicht nicht aus, um die nötige Verbindungsfahrt durchzuführen.
				pnItem = DurationInSeconds::invalid();
				continue;
			}

			pnItem = intervall - emptyTripDuration;
		}
	}

	_pnMatrix->copyToDevice();

	stopwatch.stop("PrevNextMatrix erzeugt (CPU): ");
}


const int BlockSize = 32;

__global__ void createKernel(CuMatrix1<DurationInSeconds> *pnMatrix, CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips,  
	ConnectionMatrix *connectionMatrix,	int numOfServiceTrips)
{
	assert(blockDim.x <= BlockSize);
	assert(blockDim.y <= BlockSize);

	__shared__ CuServiceTrip prevServiceTrips[BlockSize];
	__shared__ CuServiceTrip nextServiceTrips[BlockSize];

	__syncthreads(); // CuServiceTrip enthält Member mit nicht-trivialen Konstruktoren --> Race-conditions!

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	ServiceTripId row(blockIdx.x * blockDim.x + tx);
	ServiceTripId col(blockIdx.y * blockDim.y + ty);

	// Shared memory befüllen:
	if (ty == 0 && (short)row < numOfServiceTrips) {
		prevServiceTrips[tx] = serviceTrips->getServiceTrip(row);
	}

	if (tx == 0 && (short)col < numOfServiceTrips) {
		nextServiceTrips[ty] = serviceTrips->getServiceTrip(col);
	}

	__syncthreads();

	if ((short)row >= numOfServiceTrips || (short)col >= numOfServiceTrips) return;

	DurationInSeconds &pnItem = pnMatrix->itemAt((short)row, (short)col);

	if (row == col) {
		// Eine Servicefahrt kann natürlich nicht vor oder nach sich selbst durchgeführt werden.
		pnItem = DurationInSeconds::invalid();
		return;
	}

	CuServiceTrip prevServTrip = prevServiceTrips[tx];
	CuServiceTrip nextServTrip = nextServiceTrips[ty];

	if (prevServTrip.arrival >= nextServTrip.departure) 
	{
		// Fahrt A endet erst, nachdem Fahrt B begonnen hat.
		pnItem = DurationInSeconds::invalid();
		return;
	}

	DurationInSeconds emptyTripDuration(0);
 	if (prevServTrip.toStopId != nextServTrip.fromStopId) {
		// Es ist eine Verbindungsfahrt nötig, um von der Endhaltestelle der Fahrt A zu Starthaltestelle der Fahrt B zu kommen.
		EmptyTripId et_id = connectionMatrix->getEmptyTripId(prevServTrip.toStopId, nextServTrip.fromStopId);
		if (!et_id.isValid()) {
			// Die nötige Verbingungsfahrt existiert nicht.
			pnItem = DurationInSeconds::invalid();
			return;
		}
		else {
			emptyTripDuration = emptyTrips->getEmptyTrip(et_id).duration;
		}
	}

	DurationInSeconds intervall = DurationInSeconds((int)nextServTrip.departure - (int)prevServTrip.arrival - 2);

	if (intervall < emptyTripDuration) {
		// Der Zeitraum zwischen den beiden Servicefahrten reicht nicht aus, um die nötige Verbindungsfahrt durchzuführen.
		pnItem = DurationInSeconds::invalid();
		return;
	}

	pnItem = intervall - emptyTripDuration;
}


void PrevNextMatrix::createGpu(CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix, int numOfServiceTrips)
{
	Stopwatch stopwatch;
	stopwatch.start();
	
	int numOfBlocks = (numOfServiceTrips + BlockSize - 1) / BlockSize;
	dim3 dimGrid(numOfBlocks, numOfBlocks);
	dim3 dimBlock(BlockSize, BlockSize);

	createKernel << <dimGrid, dimBlock >> > (_pnMatrix->getDevPtr(), serviceTrips->getDevPtr(), emptyTrips->getDevPtr(), 
		connectionMatrix->getDevPtr(), numOfServiceTrips);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	_pnMatrix->copyToHost();

	stopwatch.stop("PrevNextMatrix erzeugt (GPU): ");
}


shared_ptr<CuVector1<ServiceTripId>> PrevNextMatrix::collectSuccessors(ServiceTripId trip) const
{
	// TODO Spezialisierte Matrix verwenden, in der nur die möglichen Nachfolger gelistet sind.
	shared_ptr<CuVector1<ServiceTripId>> retVal;
	int numOfCols = _pnMatrix->getNumOfCols();
	int numOfSuccessors = 0;
	CuVector1<ServiceTripId> temp(numOfCols);
	for (ServiceTripId s(0); s < numOfCols; s++) {
		if (checkCombination(trip, s)) {
			temp.set(numOfSuccessors++, s);
		}
	}
	if (numOfSuccessors > 0) {
		retVal = make_shared<CuVector1<ServiceTripId>>(numOfSuccessors);
		for (int i = 0; i < numOfSuccessors; i++) {
			retVal->set(i, temp.get(i));
		}
	}
	return retVal;
}


shared_ptr<CuVector1<ServiceTripId>> PrevNextMatrix::collectPredecessors(ServiceTripId trip) const
{
	// TODO Spezialisierte Matrix verwenden, in der nur die möglichen Vorgänger gelistet sind.
	shared_ptr<CuVector1<ServiceTripId>> retVal;
	int numOfRows = _pnMatrix->getNumOfRows();
	int numOfPredecessors = 0;
	CuVector1<ServiceTripId> temp(numOfRows);
	for (ServiceTripId s(0); s < numOfRows; s++) {
		if (checkCombination(s, trip)) {
			temp.set(numOfPredecessors++, s);
		}
	}
	if (numOfPredecessors > 0) {
		retVal = make_shared<CuVector1<ServiceTripId>>(numOfPredecessors);
		for (int i = 0; i < numOfPredecessors; i++) {
			retVal->set(i, temp.get(i));
		}
	}
	return retVal;
}


bool PrevNextMatrix::check(CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix)
{
	Stopwatch stopwatch;
	stopwatch.start();

	for (ServiceTripId tripA_id = ServiceTripId(0); tripA_id < ServiceTripId(_pnMatrix->getNumOfRows()); tripA_id++) {
		for (ServiceTripId tripB_id = ServiceTripId(0); tripB_id < ServiceTripId(_pnMatrix->getNumOfCols()); tripB_id++) {
			DurationInSeconds value = _pnMatrix->get((short)tripA_id, (short)tripB_id);
			if (tripA_id == tripB_id) {
				if (value.isValid()) {
					std::cerr << "PrevNextMatrix::check ERROR: tripA_id=" << (short)tripA_id << " tripB_id=" << (short)tripB_id << std::endl;
					return false;
				}
			}
			else {
				const CuServiceTrip &servTripA = serviceTrips->getServiceTrip(tripA_id);
				const CuServiceTrip &servTripB = serviceTrips->getServiceTrip(tripB_id);

				DurationInSeconds intervall = DurationInSeconds((int)servTripB.departure - (int)servTripA.arrival - 2);
				DurationInSeconds emtpyTripDuration = DurationInSeconds(0);
				if (intervall <= DurationInSeconds(0)) {
					intervall = DurationInSeconds::invalid();
					emtpyTripDuration = DurationInSeconds::invalid();
				}
				else {
					if (servTripA.toStopId != servTripB.fromStopId) {
						EmptyTripId emptyTripId = connectionMatrix->getEmptyTripId(servTripA.toStopId, servTripB.fromStopId);

						if (!emptyTripId.isValid()) {
							if (value.isValid()) {
								std::cerr << "PrevNextMatrix::check ERROR: Keine Verbindungsfahrt zwischen tripA_id=" 
									<< (short)tripA_id << " und tripB_id=" << (short)tripB_id << std::endl;
								return false;
							}
							intervall = DurationInSeconds::invalid();
							emtpyTripDuration = DurationInSeconds::invalid();
						}
						else {
							emtpyTripDuration = emptyTrips->getEmptyTrip(emptyTripId).duration;
							if (intervall < emtpyTripDuration) {
								intervall = DurationInSeconds::invalid();
								emtpyTripDuration = DurationInSeconds::invalid();
							}
						}
					}
				}

				if (intervall.isValid()) {
					if (value != intervall - emtpyTripDuration) {
						std::cerr << "PrevNextMatrix::check ERROR: value!=netIntervall tripA_id=" << (short)tripA_id 
							<< " tripB_id=" << (short)tripB_id << std::endl;
						return false;
					}
				}
				else {
					if (value.isValid()) {
						std::cerr << "PrevNextMatrix::check ERROR: value!=-1 tripA_id=" << (short)tripA_id << " tripB_id=" 
							<< (short)tripB_id << std::endl;
						return false;
					}
				}
			}

		}
	}

	stopwatch.stop("PrevNextMatrix geprüft: ");

	return true;
}
