#include "CuServiceTrips.h"

#include "EvspLimits.h"

#include "CudaCheck.h"

#include <stdexcept> 
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h> 

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"


using namespace std;



CuServiceTrips::CuServiceTrips(int maxNumOfServiceTrips)
	: _numOfServiceTrips(0), _devicePtr(0), _earliestStartTime(INT_MAX), _latestStartTime(0)

{
	if (maxNumOfServiceTrips > Max_ServiceTrips) throw new std::invalid_argument("maxNumOfServiceTrips > Max_ServiceTrips");
	_serviceTrips = new CuVector1<CuServiceTrip>(maxNumOfServiceTrips);
}


CuServiceTrips::~CuServiceTrips()
{
	if (_serviceTrips) delete _serviceTrips;
	if (_devicePtr)
	{
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


CuServiceTrips* CuServiceTrips::getDevPtr()
{
	if (!_devicePtr)
	{
		CuServiceTrips temp;
		temp._devicePtr = 0;
		temp._numOfServiceTrips = _numOfServiceTrips;
		temp._serviceTrips = _serviceTrips->getDevPtr();
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuServiceTrips)));
		
		CUDA_CHECK(cudaMemcpy(_devicePtr, &temp, sizeof(CuServiceTrips), cudaMemcpyHostToDevice));
		temp._serviceTrips = 0;
	}
	return _devicePtr;
}


void CuServiceTrips::add(StopId fromStopId, StopId toStopId, DistanceInMeters distance, VehicleTypeGroupId vehicleTypeGroupId, PointInTime departure, PointInTime arrival)
{
	if (_numOfServiceTrips >= Max_ServiceTrips) throw std::runtime_error("Die maximale Anzahl von Servicefahrten ist überschritten!");

	assert(fromStopId < StopId(Max_Stops));
	assert(toStopId < StopId(Max_Stops));
	assert(vehicleTypeGroupId < VehicleTypeGroupId(Max_VehicleTypeGroups));
	assert(distance > DistanceInMeters(0)); // Die Strecke einer Servicefahrt sollte länger als 0 Meter sein.
	assert(distance < DistanceInMeters(1000000)); // Eine Servicefahrt von 1.000 km Länge ist nicht plausibel.
	assert(departure < arrival); // Eine Servicefahrt sollte länger als 0 Sekunden dauern und nicht ankommen, bevor sie gestartet ist.
	assert(arrival - departure < PointInTime(86400)); // Eine Servicefahrt, die 24 Stunden dauert ist nicht plausibel.

	CuServiceTrip &nt = (*_serviceTrips)[_numOfServiceTrips];
	nt.fromStopId = fromStopId;
	nt.toStopId = toStopId;
	nt.distance = distance;
	nt.vehicleTypeGroupId = vehicleTypeGroupId;
	nt.departure = departure;
	nt.arrival = arrival;

	if (departure < _earliestStartTime) _earliestStartTime = departure;
	if (arrival > _latestStartTime) 	_latestStartTime = arrival;

	_numOfServiceTrips++;
}


CU_HSTDEV int CuServiceTrips::getNumOfServiceTrips() const
{
	return _numOfServiceTrips;
}


CU_HSTDEV const CuServiceTrip& CuServiceTrips::getServiceTrip(ServiceTripId serviceTripId) const
{
	assert(serviceTripId.isValid());
	short s_id = (short)serviceTripId;
	assert(s_id < _numOfServiceTrips);
	return (*_serviceTrips)[s_id];
}
