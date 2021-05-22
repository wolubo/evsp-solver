#include "CuEmptyTrips.h"

#include "CudaCheck.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h> 
#include <stdexcept> 

#include "EvspLimits.h"


CuEmptyTrips::CuEmptyTrips(int maxNumOfEmptyTrips)
	: _numOfEmptyTrips(0), _devicePtr(0)
{
	if (maxNumOfEmptyTrips > Max_EmptyTrips) throw new std::invalid_argument("maxNumOfEmptyTrips > Max_EmptyTrips");
	_emptyTrips = new CuVector1<CuEmptyTrip>(maxNumOfEmptyTrips);
}


CuEmptyTrips::~CuEmptyTrips()
{
	if (_emptyTrips) delete _emptyTrips;
	if (_devicePtr)
	{
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


CuEmptyTrips* CuEmptyTrips::getDevPtr()
{
	if (!_devicePtr)
	{
		CuEmptyTrips temp;
		temp._devicePtr = 0;
		temp._numOfEmptyTrips = _numOfEmptyTrips;
		temp._emptyTrips = _emptyTrips->getDevPtr();
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuEmptyTrips)));
		
		CUDA_CHECK(cudaMemcpy(_devicePtr, &temp, sizeof(CuEmptyTrips), cudaMemcpyHostToDevice));
		temp._emptyTrips = 0;
	}
	return _devicePtr;
}


void CuEmptyTrips::add(StopId from, StopId to, DistanceInMeters dist, DurationInSeconds rt)
{
	if (_numOfEmptyTrips >= _emptyTrips->getSize()) throw std::runtime_error("Die maximale Anzahl von Verbindungsfahrten ist überschritten!");

	assert(from < StopId(Max_Stops));
	assert(to < StopId(Max_Stops));
	assert(from != to);
	assert(dist > DistanceInMeters(0)); // Die Strecke einer Verbindungsfahrt sollte länger als 0 Meter sein.
	assert(dist < DistanceInMeters(1000000)); // Eine Verbindungsfahrt von 1.000 km Länge ist nicht plausibel.
	assert(rt > DurationInSeconds(0)); // Eine Verbindungsfahrt sollte länger als 0 Sekunden dauern.
	assert(rt < DurationInSeconds(86400)); // Eine Verbindungsfahrt, die 24 Stunden dauert ist nicht plausibel.

	CuEmptyTrip &nt = (*_emptyTrips)[_numOfEmptyTrips];
	nt.fromStopId = from;
	nt.toStopId = to;
	nt.distance = dist;
	nt.duration = rt;

	//#pragma warning( push )  
	//#pragma warning( disable : 4267 ) // "Konvertierung von XXX nach YYY, Datenverlust möglich"
	//				_fromStopId[_numOfEmptyTrips] = from;
	//				_toStopId[_numOfEmptyTrips] = to;
	//				_distance[_numOfEmptyTrips] = dist;
	//				_duration[_numOfEmptyTrips] = rt;
	//#pragma warning( pop )  

	_numOfEmptyTrips++;
}


CU_HSTDEV int CuEmptyTrips::getNumOfEmptyTrips() const
{
	return _numOfEmptyTrips;
}


CU_HSTDEV const CuEmptyTrip& CuEmptyTrips::getEmptyTrip(EmptyTripId id) const
{
	assert(id.isValid());
	short s_id = (short)id;
	assert(s_id < _numOfEmptyTrips);
	return (*_emptyTrips)[s_id];
}

