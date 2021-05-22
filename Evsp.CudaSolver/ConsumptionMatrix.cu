#include "ConsumptionMatrix.h"
#include "EVSP.BaseClasses/Stopwatch.h"

#include <algorithm>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"


ConsumptionMatrix::ConsumptionMatrix(CuEmptyTrips &emptyTrips, CuServiceTrips &serviceTrips, CuVehicleTypes &vehicleTypes,
	PlattformConfig plattform)
	: _devicePtr(0)
{
	int numOfVehicleTypes = vehicleTypes.getNumOfVehicleTypes();
	int numOfServiceTrips = serviceTrips.getNumOfServiceTrips();
	int numOfEmptyTrips = emptyTrips.getNumOfEmptyTrips();

	_emptyTripConsumption = new CuMatrix1<KilowattHour>(numOfEmptyTrips, numOfVehicleTypes);
	//_emptyTripConsumption->setAll(KilowattHour(-1.0f));

	_serviceTripConsumption = new CuMatrix1<KilowattHour>(numOfServiceTrips, numOfVehicleTypes);
	//_serviceTripConsumption->setAll(KilowattHour(-1.0f));

	switch (plattform) {
	case PlattformConfig::CPU:
		createOnCpu(emptyTrips, numOfEmptyTrips, serviceTrips, numOfServiceTrips, vehicleTypes, numOfVehicleTypes);
		break;
	case PlattformConfig::GPU:
		createOnGpu(emptyTrips, numOfEmptyTrips, serviceTrips, numOfServiceTrips, vehicleTypes, numOfVehicleTypes);
		break;
	default:
		assert(false);
	}
}


ConsumptionMatrix::~ConsumptionMatrix()
{
	if (_emptyTripConsumption) delete _emptyTripConsumption;
	if (_serviceTripConsumption) delete _serviceTripConsumption;
	if (_devicePtr)
	{
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


ConsumptionMatrix* ConsumptionMatrix::getDevPtr()
{
	if (!_devicePtr)
	{
		ConsumptionMatrix *temp_devicePtr = 0;
		CuMatrix1<KilowattHour> *temp_emptyTripConsumption = _emptyTripConsumption;
		CuMatrix1<KilowattHour> *temp_serviceTripConsumption = _serviceTripConsumption;

		_emptyTripConsumption = _emptyTripConsumption->getDevPtr();
		_serviceTripConsumption = _serviceTripConsumption->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&temp_devicePtr, sizeof(ConsumptionMatrix)));
		CUDA_CHECK(cudaMemcpy(temp_devicePtr, this, sizeof(ConsumptionMatrix), cudaMemcpyHostToDevice));

		_emptyTripConsumption = temp_emptyTripConsumption;
		_serviceTripConsumption = temp_serviceTripConsumption;
		_devicePtr = temp_devicePtr;
	}
	return _devicePtr;
}


CU_HSTDEV KilowattHour ConsumptionMatrix::getEmptyTripConsumption(EmptyTripId emptyTripId, VehicleTypeId vehTypeId)
{
	return _emptyTripConsumption->get((short)emptyTripId, (short)vehTypeId);
}


CU_HSTDEV KilowattHour ConsumptionMatrix::getServiceTripConsumption(ServiceTripId servTripId, VehicleTypeId vehTypeId)
{
	return _serviceTripConsumption->get((short)servTripId, (short)vehTypeId);
}


void ConsumptionMatrix::createOnCpu(CuEmptyTrips &emptyTrips, int numOfEmptyTrips, CuServiceTrips &serviceTrips, int numOfServiceTrips,
	CuVehicleTypes &vehicleTypes, int numOfVehicleTypes)
{
	Stopwatch stopwatch;
	stopwatch.start();

	float value;
	for (VehicleTypeId vtId(0); vtId < numOfVehicleTypes; vtId++) {
		CuVehicleType vt = vehicleTypes.getVehicleType(VehicleTypeId(vtId));
		for (ServiceTripId stId(0); stId < numOfServiceTrips; stId++) {
			value = (float)vt.getServiceTripConsumption(serviceTrips.getServiceTrip(ServiceTripId(stId)).distance);
			_serviceTripConsumption->set((short)stId, (short)vtId, KilowattHour(value));
		}
		for (EmptyTripId etId(0); etId < numOfEmptyTrips; etId++) {
			value = (float)vt.getEmptyTripConsumption(emptyTrips.getEmptyTrip(EmptyTripId(etId)).distance);
			_emptyTripConsumption->set((short)etId, (short)vtId, KilowattHour(value));
		}
	}
	_emptyTripConsumption->copyToDevice();
	_serviceTripConsumption->copyToDevice();

	stopwatch.stop("ConsumptionMatrix erzeugt (CPU): ");
}


__global__ void computeConsumptionKernel(int numOfVehicleTypes, int numOfServiceTrips, int numOfEmptyTrips,
	CuEmptyTrips *emptyTrips, CuServiceTrips *serviceTrips, CuVehicleTypes *vehicleTypes,
	CuMatrix1<KilowattHour> *emptyTripConsumptionBuffer, CuMatrix1<KilowattHour> *serviceTripConsumptionBuffer)
{
	int tripId = blockIdx.x * blockDim.x + threadIdx.x;
	int vehId = blockIdx.y * blockDim.y + threadIdx.y;

	if (vehId >= numOfVehicleTypes) {
		return;
	}

	float value;

	CuVehicleType vt = vehicleTypes->getVehicleType(VehicleTypeId(vehId));

	if (tripId < numOfServiceTrips) {
		value = (float)vt.getServiceTripConsumption(serviceTrips->getServiceTrip(ServiceTripId(tripId)).distance);
		serviceTripConsumptionBuffer->set(tripId, vehId, KilowattHour(value));
	}

	if (tripId < numOfEmptyTrips) {
		value = (float)vt.getEmptyTripConsumption(emptyTrips->getEmptyTrip(EmptyTripId(tripId)).distance);
		emptyTripConsumptionBuffer->set(tripId, vehId, KilowattHour(value));
	}
}


void ConsumptionMatrix::createOnGpu(CuEmptyTrips &emptyTrips, int numOfEmptyTrips, CuServiceTrips &serviceTrips, int numOfServiceTrips,
	CuVehicleTypes &vehicleTypes, int numOfVehicleTypes)
{
	Stopwatch stopwatch;
	stopwatch.start();

	int numOfThreads_x = std::max(numOfServiceTrips, numOfEmptyTrips);
	int numOfThreads_y = numOfVehicleTypes;

	const int blockSize_x = 256;
	const int blockSize_y = 4;

	int numOfBlocks_x = (numOfThreads_x + blockSize_x - 1) / blockSize_x;
	int numOfBlocks_y = (numOfThreads_y + blockSize_y - 1) / blockSize_y;

	dim3 dimGrid(numOfBlocks_x, numOfBlocks_y);
	dim3 dimBlock(blockSize_x, blockSize_y);
	computeConsumptionKernel << <dimGrid, dimBlock >> > (numOfVehicleTypes, numOfServiceTrips, numOfEmptyTrips,
		emptyTrips.getDevPtr(), serviceTrips.getDevPtr(), vehicleTypes.getDevPtr(),
		_emptyTripConsumption->getDevPtr(), _serviceTripConsumption->getDevPtr());
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	_emptyTripConsumption->copyToHost();
	_serviceTripConsumption->copyToHost();

	stopwatch.stop("ConsumptionMatrix erzeugt (GPU): ");
}