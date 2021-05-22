#include "ServiceTripCostMatrix.h"
#include <iostream>
#include <stdexcept> 
#include <assert.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/Stopwatch.h"
#include "CudaCheck.h"
#include "RandomGpu.h"
#include "EvspLimits.h"


ServiceTripCostMatrix::ServiceTripCostMatrix(CuServiceTrips *serviceTrips, CuVehicleTypes *vehicleTypes,
	CuVehicleTypeGroups *vehicleTypeGroups, PlattformConfig plattform)
	: _devicePtr(0), _distanceDependentCosts(0), _timeDependentCosts(0), _bestVehicleType(0), _consumption(0)
{
	int numOfServiceTrips = serviceTrips->getNumOfServiceTrips();
	int numOfVehicleTypes = vehicleTypes->getNumOfVehicleTypes();
	_distanceDependentCosts = new CuMatrix1<AmountOfMoney>(numOfServiceTrips, numOfVehicleTypes);
	_timeDependentCosts = new CuMatrix1<AmountOfMoney>(numOfServiceTrips, numOfVehicleTypes);
	_consumption = new CuMatrix1<KilowattHour>(numOfServiceTrips, numOfVehicleTypes);
	_bestVehicleType = new CuVector1<VehicleTypeId>(numOfServiceTrips);

	switch (plattform) {
	case PlattformConfig::CPU:
		createCpu(serviceTrips, numOfServiceTrips, vehicleTypes,  numOfVehicleTypes, vehicleTypeGroups);
		break;
	case PlattformConfig::GPU:
		createGpu(serviceTrips, numOfServiceTrips, vehicleTypes,  numOfVehicleTypes, vehicleTypeGroups);
		break;
	default:
		assert(false);
	}
}


ServiceTripCostMatrix::~ServiceTripCostMatrix()
{
	if (_distanceDependentCosts) delete _distanceDependentCosts;
	if (_timeDependentCosts) delete _timeDependentCosts;
	if (_consumption) delete _consumption;
	if (_bestVehicleType) delete _bestVehicleType;

	if (_devicePtr)
	{
#ifdef __CUDACC__
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
#endif
	}
}


ServiceTripCostMatrix* ServiceTripCostMatrix::getDevPtr()
{
	if (!_devicePtr) {
		CuMatrix1<AmountOfMoney> *temp_distanceDependentCosts = _distanceDependentCosts;
		CuMatrix1<AmountOfMoney> *temp_timeDependentCosts = _timeDependentCosts;
		CuMatrix1<KilowattHour> *temp_consumption = _consumption;
		CuVector1<VehicleTypeId> *temp_bestVehicleType = _bestVehicleType;
		ServiceTripCostMatrix* tempDevicePtr = 0;

		_distanceDependentCosts = _distanceDependentCosts->getDevPtr();
		_timeDependentCosts = _timeDependentCosts->getDevPtr();
		_consumption = _consumption->getDevPtr();
		_bestVehicleType = _bestVehicleType->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempDevicePtr, sizeof(ServiceTripCostMatrix)));
		CUDA_CHECK(cudaMemcpy(tempDevicePtr, this, sizeof(ServiceTripCostMatrix), cudaMemcpyHostToDevice));

		_distanceDependentCosts = temp_distanceDependentCosts;
		_timeDependentCosts = temp_timeDependentCosts;
		_consumption = temp_consumption;
		_bestVehicleType = temp_bestVehicleType;
		_devicePtr = tempDevicePtr;
	}
	return _devicePtr;
}


void ServiceTripCostMatrix::copyToHost()
{
	if (_devicePtr) {
		_distanceDependentCosts->copyToHost();
		_timeDependentCosts->copyToHost();
		_consumption->copyToHost();
		_bestVehicleType->copyToHost();
	}
}


void ServiceTripCostMatrix::createCpu(CuServiceTrips *serviceTrips, int numOfServiceTrips, CuVehicleTypes *vehicleTypes, int numOfVehicleTypes,
	CuVehicleTypeGroups *vehicleTypeGroups)
{
	Stopwatch stopwatch;
	stopwatch.start();

	for (ServiceTripId stId(0); stId < numOfServiceTrips; stId++) {
		const CuServiceTrip &serviceTrip = serviceTrips->getServiceTrip(stId);
		AmountOfMoney bestTotalCost = AmountOfMoney(INT_MAX);
		VehicleTypeId theBest;
		for (VehicleTypeId vtId(0); vtId < numOfVehicleTypes; vtId++) {
			const CuVehicleType &vehicleType = vehicleTypes->getVehicleType(vtId);
			if (vehicleTypeGroups->hasVehicleType(serviceTrip.vehicleTypeGroupId, vtId))
			{
				AmountOfMoney distanceDependent = vehicleType.getDistanceDependentCosts(serviceTrip.distance);
				_distanceDependentCosts->set((short)stId, (short)vtId, distanceDependent);
				AmountOfMoney timeDependent = vehicleType.getTimeDependentCosts(serviceTrip.getDuration());
				_timeDependentCosts->set((short)stId, (short)vtId, timeDependent);
				_consumption->set((short)stId, (short)vtId, vehicleType.getServiceTripConsumption(serviceTrip.distance));

				AmountOfMoney tc = distanceDependent + timeDependent;
				if (tc < bestTotalCost) {
					bestTotalCost = tc;
					theBest = vtId;
				}
			}
			else {
				_distanceDependentCosts->set((short)stId, (short)vtId, AmountOfMoney::invalid());
				_timeDependentCosts->set((short)stId, (short)vtId, AmountOfMoney::invalid());
				_consumption->set((short)stId, (short)vtId, KilowattHour(0.0f));
			}
		}
		assert(theBest.isValid());
		_bestVehicleType->set((short)stId, theBest);
	}

	stopwatch.stop("ServiceTripCostMatrix erzeugt (CPU): ");
}

const int BlockSize_x = 256;
const int BlockSize_y = 4;

__global__ void createKernel(CuMatrix1<AmountOfMoney> *distanceDependentCosts, CuMatrix1<AmountOfMoney> *timeDependentCosts, 
	CuMatrix1<KilowattHour> *consumption, CuServiceTrips *serviceTrips, CuVehicleTypes *vehicleTypes,
	CuVehicleTypeGroups *vehicleTypeGroups, int numOfServiceTrips, int numOfVehicleTypes)
{
	__shared__ CuServiceTrip serviceTrip_s[BlockSize_x];
	__shared__ CuVehicleType vehicleType_s[BlockSize_y];

	__syncthreads(); // Einige Member haben nicht-trivialen Konstruktoren --> Race-conditions!
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	ServiceTripId stId(blockIdx.x * blockDim.x + tx);
	VehicleTypeId vtId(blockIdx.y * blockDim.y + ty);

	// Shared memory befüllen:
	if (ty == 0 && (short)stId < numOfServiceTrips) {
		serviceTrip_s[tx] = serviceTrips->getServiceTrip(stId);
	}
	if (tx == 0 && (short)vtId < numOfVehicleTypes) {
		vehicleType_s[ty] = vehicleTypes->getVehicleType(vtId);
	}
	__syncthreads();

	if ((short)stId >= numOfServiceTrips || (short)vtId >= numOfVehicleTypes) return;

	CuServiceTrip &serviceTrip = serviceTrip_s[tx];
	CuVehicleType &vehicleType = vehicleType_s[ty];

	if (vehicleTypeGroups->hasVehicleType(serviceTrip.vehicleTypeGroupId, vtId))
	{
		distanceDependentCosts->set((short)stId, (short)vtId, vehicleType.getDistanceDependentCosts(serviceTrip.distance));
		timeDependentCosts->set((short)stId, (short)vtId, vehicleType.getTimeDependentCosts(serviceTrip.getDuration()));
		consumption->set((short)stId, (short)vtId, vehicleType.getServiceTripConsumption(serviceTrip.distance));
	}
	else {
		distanceDependentCosts->set((short)stId, (short)vtId, AmountOfMoney::invalid());
		timeDependentCosts->set((short)stId, (short)vtId, AmountOfMoney::invalid());
		consumption->set((short)stId, (short)vtId, KilowattHour(0.0f));
	}
}


void ServiceTripCostMatrix::startCreateKernel(int numOfServiceTrips, int numOfVehicleTypes, 
	CuServiceTrips *serviceTrips, CuVehicleTypes *vehicleTypes, CuVehicleTypeGroups *vehicleTypeGroups)
{
	int numOfBlocks_x = (numOfServiceTrips + BlockSize_x - 1) / BlockSize_x;
	int numOfBlocks_y = (numOfVehicleTypes + BlockSize_y - 1) / BlockSize_y;
	dim3 dimGrid(numOfBlocks_x, numOfBlocks_y);
	dim3 dimBlock(BlockSize_x, BlockSize_y);

	createKernel << <dimGrid, dimBlock >> >(_distanceDependentCosts->getDevPtr(), _timeDependentCosts->getDevPtr(), _consumption->getDevPtr(),
		serviceTrips->getDevPtr(), vehicleTypes->getDevPtr(), vehicleTypeGroups->getDevPtr(), numOfServiceTrips, numOfVehicleTypes);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	_distanceDependentCosts->copyToHost();
	_timeDependentCosts->copyToHost();
	_consumption->copyToHost();
}


__global__ void findBestVehTypeKernel(CuMatrix1<AmountOfMoney> *distanceDependentCosts, CuMatrix1<AmountOfMoney> *timeDependentCosts,
	CuVector1<VehicleTypeId> *bestVehicleType, int numOfServiceTrips, int numOfVehicleTypes)
{
	int tx = blockIdx.x;
	ServiceTripId stId(tx * blockDim.x + threadIdx.x);

	// Suche den Fahrzeugtypen, der die jeweilige Servicefahrt zum günstigsten Gesamtpreis durchführen kann.
	if ((short)stId < numOfServiceTrips) {
		AmountOfMoney tc;
		AmountOfMoney bestTotalCost = AmountOfMoney(INT_MAX);
		VehicleTypeId theBest;
		for (VehicleTypeId bvtId(0); bvtId<numOfVehicleTypes; bvtId++) {
			tc = timeDependentCosts->get((short)stId, (short)bvtId) + distanceDependentCosts->get((short)stId, (short)bvtId);
			if (tc < bestTotalCost) {
				bestTotalCost = tc;
				theBest = bvtId;
			}
		}
		assert(theBest.isValid());
		bestVehicleType->set((short)stId, theBest);
	}
}


void ServiceTripCostMatrix::startFindBestVehTypeKernel(int numOfServiceTrips, int numOfVehicleTypes)
{
	int numOfBlocks = (numOfServiceTrips + 1024 - 1) / 1024;
	dim3 dimGrid(numOfBlocks);
	dim3 dimBlock(1024);

	findBestVehTypeKernel<< <dimGrid, dimBlock >> >(_distanceDependentCosts->getDevPtr(), _timeDependentCosts->getDevPtr(),
		_bestVehicleType->getDevPtr(), numOfServiceTrips, numOfVehicleTypes);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	_bestVehicleType->copyToHost();
}


void ServiceTripCostMatrix::createGpu(CuServiceTrips *serviceTrips, int numOfServiceTrips, CuVehicleTypes *vehicleTypes, int numOfVehicleTypes,
	CuVehicleTypeGroups *vehicleTypeGroups)
{
	Stopwatch stopwatch;
	stopwatch.start();

	startCreateKernel(numOfServiceTrips, numOfVehicleTypes, serviceTrips, vehicleTypes, vehicleTypeGroups);
	startFindBestVehTypeKernel(numOfServiceTrips, numOfVehicleTypes);

	stopwatch.stop("ServiceTripCostMatrix erzeugt: ");
}


CU_HSTDEV AmountOfMoney ServiceTripCostMatrix::getDistanceDependentCosts(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const
{
	return _distanceDependentCosts->get((short)serviceTripId, (short)vehicleTypeId);
}


CU_HSTDEV AmountOfMoney ServiceTripCostMatrix::getTimeDependentCosts(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const
{
	return _timeDependentCosts->get((short)serviceTripId, (short)vehicleTypeId);
}


CU_HSTDEV AmountOfMoney ServiceTripCostMatrix::getTotalCost(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const
{
	return getDistanceDependentCosts(serviceTripId, vehicleTypeId) + getTimeDependentCosts(serviceTripId, vehicleTypeId);
}


CU_HSTDEV KilowattHour ServiceTripCostMatrix::getConsumption(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const
{
	return _consumption->get((short)serviceTripId, (short)vehicleTypeId);
}


CU_HSTDEV bool ServiceTripCostMatrix::combinationIsValid(ServiceTripId serviceTripId, VehicleTypeId vehicleTypeId) const
{
	return getDistanceDependentCosts(serviceTripId, vehicleTypeId).isValid();
}


CU_HSTDEV VehicleTypeId ServiceTripCostMatrix::getBestVehicleType(ServiceTripId serviceTripId) const
{
	return _bestVehicleType->get((short)serviceTripId);
}