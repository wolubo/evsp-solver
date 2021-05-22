#include "CuStops.h"

#include <stdexcept> 
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h> 

#include "EvspLimits.h"

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "CudaCheck.h"



CuStops::CuStops(int maxNumOfStops, int maxNumOfDepots, int maxNumOfChargingStations)
	: _numOfStops(0), _numOfDepots(0), _numOfChargingStations(0), _devicePtr(0), _chargingStationIds(0)
{
	if (maxNumOfStops > Max_Stops) throw new std::invalid_argument("maxNumOfStops > Max_Stops");
	if (maxNumOfDepots > Max_Depots) throw new std::invalid_argument("maxNumOfDepots > Max_Depots");
	if (maxNumOfChargingStations > Max_ChargingStations) throw new std::invalid_argument("maxNumOfChargingStations > Max_ChargingStations");
	_stops = new CuVector1<CuStop>(maxNumOfStops);
	_depotIds = new CuVector1<StopId>(maxNumOfDepots);
	if (maxNumOfChargingStations > 0) {
		_chargingStationIds = new CuVector1<StopId>(maxNumOfChargingStations);
	}
}


CuStops::~CuStops()
{
	if (_stops) delete _stops;
	if (_depotIds) delete _depotIds;
	if (_chargingStationIds) delete _chargingStationIds;
	if (_devicePtr)
	{
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


CuStops* CuStops::getDevPtr()
{
	if (!_devicePtr)
	{
		CuStops temp;
		temp._devicePtr = 0;
		temp._numOfStops = _numOfStops;
		temp._numOfDepots = _numOfDepots;
		temp._numOfChargingStations = _numOfChargingStations;
		temp._stops = _stops->getDevPtr();
		temp._depotIds = _depotIds->getDevPtr();
		if (_chargingStationIds) temp._chargingStationIds = _chargingStationIds->getDevPtr();
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuStops)));
		
		CUDA_CHECK(cudaMemcpy(_devicePtr, &temp, sizeof(CuStops), cudaMemcpyHostToDevice));
		temp._stops = 0;
		temp._depotIds = 0;
		temp._chargingStationIds = 0;
	}
	return _devicePtr;
}


void CuStops::add(bool depot, bool charging)
{
	if (_numOfStops >= _stops->getSize()) throw std::runtime_error("Die maximale Anzahl von Haltestellen ist überschritten!");

	CuStop &stop = (*_stops)[_numOfStops];
	stop.isDepot = depot;
	stop.isChargingStation = charging;

	if (depot) {
		if (_numOfDepots >= _depotIds->getSize()) throw std::runtime_error("Die maximale Depotanzahl ist überschritten!");
		(*_depotIds)[_numOfDepots] = StopId(_numOfStops);
		_numOfDepots++;
	}

	if (charging && _chargingStationIds) {
		if (_numOfChargingStations >= _chargingStationIds->getSize()) throw std::runtime_error("Die maximale Ladestationsanzahl ist überschritten!");
		(*_chargingStationIds)[_numOfChargingStations] = StopId(_numOfStops);
		_numOfChargingStations++;
	}

	_numOfStops++;
}


CU_HSTDEV int CuStops::getNumOfStops() const
{
	return _numOfStops;
}


CU_HSTDEV int CuStops::getNumOfDepots() const
{
	return _numOfDepots;
}


CU_HSTDEV int CuStops::getNumOfChargingStations() const
{
	return _numOfChargingStations;
}


CU_HSTDEV const CuStop& CuStops::getStop(StopId stopId) const
{
	assert(stopId.isValid());
	short s_id = (short)stopId;
	assert(s_id < _numOfStops);
	return (*_stops)[s_id];
}


CU_HSTDEV StopId CuStops::getStopIdOfDepot(DepotId depotId) const
{
	assert(depotId.isValid());
	assert(depotId < DepotId(_numOfDepots));
	if (depotId < DepotId(_numOfDepots)) return (*_depotIds)[(short)depotId];
	return StopId::invalid();
}


CU_HSTDEV StopId CuStops::getStopIdOfChargingStation(ChargingStationId chargingStationId) const
{
	assert(chargingStationId.isValid());
	if (!_chargingStationIds) return StopId::invalid();
	assert(chargingStationId < ChargingStationId(_numOfChargingStations));
	if (chargingStationId < ChargingStationId(_numOfChargingStations)) return (*_chargingStationIds)[(short)chargingStationId];
	return StopId::invalid();
}
