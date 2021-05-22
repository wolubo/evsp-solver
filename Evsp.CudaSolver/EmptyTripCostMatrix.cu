#include "EmptyTripCostMatrix.h"
#include <iostream>
#include <stdexcept> 
#include <assert.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/Stopwatch.h"
#include "CudaCheck.h"
#include "RandomGpu.h"
#include "EvspLimits.h"


EmptyTripCostMatrix::EmptyTripCostMatrix(CuEmptyTrips *emptyTrips, CuVehicleTypes *vehicleTypes, PlattformConfig plattform)
	: _devicePtr(0), _distanceDependentCosts(0), _timeDependentCosts(0), _consumption(0)
{
	int numOfEmptyTrips = emptyTrips->getNumOfEmptyTrips();
	int numOfVehicleTypes = vehicleTypes->getNumOfVehicleTypes();
	_distanceDependentCosts = new CuMatrix1<AmountOfMoney>(numOfEmptyTrips, numOfVehicleTypes);
	_timeDependentCosts = new CuMatrix1<AmountOfMoney>(numOfEmptyTrips, numOfVehicleTypes);
	_consumption = new CuMatrix1<KilowattHour>(numOfEmptyTrips, numOfVehicleTypes);

	switch (plattform) {
	case PlattformConfig::CPU:
		createCpu(emptyTrips, numOfEmptyTrips, vehicleTypes, numOfVehicleTypes);
		break;
	case PlattformConfig::GPU:
		createGpu(emptyTrips, numOfEmptyTrips, vehicleTypes, numOfVehicleTypes);
		break;
	default:
		assert(false);
	}
}


EmptyTripCostMatrix::~EmptyTripCostMatrix()
{
	if (_distanceDependentCosts) delete _distanceDependentCosts;
	if (_timeDependentCosts) delete _timeDependentCosts;
	if (_consumption) delete _consumption;
	if (_devicePtr)
	{
#ifdef __CUDACC__
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
#endif
	}
}


EmptyTripCostMatrix* EmptyTripCostMatrix::getDevPtr()
{
	if (!_devicePtr) {
		CuMatrix1<AmountOfMoney> *temp_distanceDependentCosts = _distanceDependentCosts;
		CuMatrix1<AmountOfMoney> *temp_timeDependentCosts = _timeDependentCosts;
		CuMatrix1<KilowattHour> *temp_consumption = _consumption;
		EmptyTripCostMatrix* tempDevicePtr = 0;

		_distanceDependentCosts = _distanceDependentCosts->getDevPtr();
		_timeDependentCosts = _timeDependentCosts->getDevPtr();
		_consumption = _consumption->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempDevicePtr, sizeof(EmptyTripCostMatrix)));
		CUDA_CHECK(cudaMemcpy(tempDevicePtr, this, sizeof(EmptyTripCostMatrix), cudaMemcpyHostToDevice));

		_distanceDependentCosts = temp_distanceDependentCosts;
		_timeDependentCosts = temp_timeDependentCosts;
		_consumption = temp_consumption;
		_devicePtr = tempDevicePtr;
	}
	return _devicePtr;
}


void EmptyTripCostMatrix::copyToHost()
{
	if (_devicePtr) {
		_distanceDependentCosts->copyToHost();
		_timeDependentCosts->copyToHost();
		_consumption->copyToHost();
	}
}


void EmptyTripCostMatrix::createCpu(CuEmptyTrips *emptyTrips, int numOfEmptyTrips, CuVehicleTypes *vehicleTypes, int numOfVehicleTypes)
{
	Stopwatch stopwatch;
	stopwatch.start();

	for (EmptyTripId etId(0); etId < numOfEmptyTrips; etId++) {
		const CuEmptyTrip &emptyTrip = emptyTrips->getEmptyTrip(etId);
		for (VehicleTypeId vtId(0); vtId < numOfVehicleTypes; vtId++) {
			const CuVehicleType &vehicleType = vehicleTypes->getVehicleType(vtId);
			_distanceDependentCosts->set((short)etId, (short)vtId, vehicleType.getDistanceDependentCosts(emptyTrip.distance));
			_timeDependentCosts->set((short)etId, (short)vtId, vehicleType.getTimeDependentCosts(emptyTrip.duration));
			_consumption->set((short)etId, (short)vtId, vehicleType.getEmptyTripConsumption(emptyTrip.distance));
		}
	}
	_distanceDependentCosts->copyToDevice();
	_timeDependentCosts->copyToDevice();
	_consumption->copyToDevice();

	stopwatch.stop("EmptyTripCostMatrix erzeugt (CPU): ");
}


const int BlockSize_x = 256;
const int BlockSize_y = 4;

__global__ void createKernel(CuMatrix1<AmountOfMoney> *distanceDependentCosts, CuMatrix1<AmountOfMoney> *timeDependentCosts,
	CuMatrix1<KilowattHour> *consumption, CuEmptyTrips *emptyTrips, CuVehicleTypes *vehicleTypes,
	int numOfEmptyTrips, int numOfVehicleTypes)
{
	__shared__ CuEmptyTrip emptyTrip_s[BlockSize_x];
	__shared__ CuVehicleType vehicleType_s[BlockSize_y];

	__syncthreads(); // Einige Member haben nicht-trivialen Konstruktoren --> Race-conditions!

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	EmptyTripId etId(blockIdx.x * blockDim.x + tx);
	VehicleTypeId vtId(blockIdx.y * blockDim.y + ty);

	// Shared memory befüllen:
	if (ty == 0 && (short)etId < numOfEmptyTrips) {
		emptyTrip_s[tx] = emptyTrips->getEmptyTrip(etId);
	}
	if (tx == 0 && (short)vtId < numOfVehicleTypes) {
		vehicleType_s[ty] = vehicleTypes->getVehicleType(vtId);
	}
	__syncthreads();

	if ((short)etId >= numOfEmptyTrips || (short)vtId >= numOfVehicleTypes) return;

	CuEmptyTrip &emptyTrip = emptyTrip_s[tx];
	CuVehicleType &vehicleType = vehicleType_s[ty];

	distanceDependentCosts->set((short)etId, (short)vtId, vehicleType.getDistanceDependentCosts(emptyTrip.distance));
	timeDependentCosts->set((short)etId, (short)vtId, vehicleType.getTimeDependentCosts(emptyTrip.duration));
	consumption->set((short)etId, (short)vtId, vehicleType.getEmptyTripConsumption(emptyTrip.distance));
}


void EmptyTripCostMatrix::createGpu(CuEmptyTrips *emptyTrips, int numOfEmptyTrips, CuVehicleTypes *vehicleTypes, int numOfVehicleTypes)
{
	Stopwatch stopwatch;
	stopwatch.start();

	int numOfBlocks_x = (numOfEmptyTrips + BlockSize_x - 1) / BlockSize_x;
	int numOfBlocks_y = (numOfVehicleTypes + BlockSize_y - 1) / BlockSize_y;
	dim3 dimGrid(numOfBlocks_x, numOfBlocks_y);
	dim3 dimBlock(BlockSize_x, BlockSize_y);

	createKernel << <dimGrid, dimBlock >> > (_distanceDependentCosts->getDevPtr(), _timeDependentCosts->getDevPtr(), _consumption->getDevPtr(),
		emptyTrips->getDevPtr(), vehicleTypes->getDevPtr(), numOfEmptyTrips, numOfVehicleTypes);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	_distanceDependentCosts->copyToHost();
	_timeDependentCosts->copyToHost();
	_consumption->copyToHost();

	stopwatch.stop("EmptyTripCostMatrix erzeugt (GPU): ");
}


CU_HSTDEV AmountOfMoney EmptyTripCostMatrix::getDistanceDependentCosts(EmptyTripId emptyTripId, VehicleTypeId vehicleTypeId) const
{
	return _distanceDependentCosts->get((short)emptyTripId, (short)vehicleTypeId);
}


CU_HSTDEV AmountOfMoney EmptyTripCostMatrix::getTimeDependentCosts(EmptyTripId emptyTripId, VehicleTypeId vehicleTypeId) const
{
	return _timeDependentCosts->get((short)emptyTripId, (short)vehicleTypeId);
}


CU_HSTDEV AmountOfMoney EmptyTripCostMatrix::getTotalCost(EmptyTripId emptyTripId, VehicleTypeId vehicleTypeId) const
{
	return getDistanceDependentCosts(emptyTripId, vehicleTypeId) + getTimeDependentCosts(emptyTripId, vehicleTypeId);
}


CU_HSTDEV KilowattHour EmptyTripCostMatrix::getConsumption(EmptyTripId emptyTripId, VehicleTypeId vehicleTypeId) const
{
	return _consumption->get((short)emptyTripId, (short)vehicleTypeId);
}
