#include "CuVehicleTypeGroups.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h> 
#include <stdexcept> 

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "CudaCheck.h"


using namespace std;


CuVehicleTypeGroups::CuVehicleTypeGroups(int maxNumOfVehicleTypeGroups, int maxNumOfVehicleTypes)
	: _numOfVehicleTypeGroups(0), _devicePtr(0)
{
	if (maxNumOfVehicleTypeGroups > Max_VehicleTypeGroups)
		throw new std::invalid_argument("maxNumOfVehicleTypeGroups > Max_VehicleTypeGroups");

	if (maxNumOfVehicleTypes > Max_VehicleTypes)
		throw new std::invalid_argument("maxNumOfVehicleTypes > Max_VehicleTypes");

	_numOfVehicleTypes = new CuVector1<ushort>(maxNumOfVehicleTypeGroups);
	_vehicleTypeIds = new CuMatrix1<VehicleTypeId>(maxNumOfVehicleTypeGroups, maxNumOfVehicleTypes);
}


CuVehicleTypeGroups::~CuVehicleTypeGroups()
{
	if (_numOfVehicleTypes) delete _numOfVehicleTypes;
	if (_vehicleTypeIds) delete _vehicleTypeIds;
	if (_devicePtr)
	{
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


CuVehicleTypeGroups* CuVehicleTypeGroups::getDevPtr()
{
	if (!_devicePtr)
	{
		CuVehicleTypeGroups temp;
		temp._devicePtr = 0;
		temp._numOfVehicleTypeGroups = _numOfVehicleTypeGroups;
		temp._vehicleTypeIds = _vehicleTypeIds->getDevPtr();
		temp._numOfVehicleTypes = _numOfVehicleTypes->getDevPtr();
		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devicePtr, sizeof(CuVehicleTypeGroups)));
		
		CUDA_CHECK(cudaMemcpy(_devicePtr, &temp, sizeof(CuVehicleTypeGroups), cudaMemcpyHostToDevice));
		temp._vehicleTypeIds = 0;
		temp._numOfVehicleTypes = 0;
	}
	return _devicePtr;
}


void CuVehicleTypeGroups::addGroup()
{
	if (_numOfVehicleTypeGroups >= _numOfVehicleTypes->getSize())
		throw runtime_error("Die maximale Anzahl von Fahrzeugtypgruppen ist überschritten!");
	(*_numOfVehicleTypes)[_numOfVehicleTypeGroups] = 0;
	_numOfVehicleTypeGroups++;
}


//#pragma warning( push )  
//#pragma warning( disable : 4267 ) // "Konvertierung von XXX nach YYY, Datenverlust möglich"
void CuVehicleTypeGroups::addType(VehicleTypeGroupId groupId, VehicleTypeId typeId)
{
	int group = (short)groupId;
	if (group >= _numOfVehicleTypeGroups) throw std::invalid_argument("groupId >= _numOfVehicleTypeGroups");
	ushort& n = (*_numOfVehicleTypes)[group];
	_vehicleTypeIds->set(group, n, typeId);
	n++;
}
//#pragma warning( pop )  


CU_HSTDEV int CuVehicleTypeGroups::getNumOfVehicleTypeGroups() const
{
	return _numOfVehicleTypeGroups;
}


CU_HSTDEV int CuVehicleTypeGroups::getNumOfVehicleTypes(VehicleTypeGroupId groupId) const
{
	assert(groupId < VehicleTypeGroupId(_numOfVehicleTypeGroups));
	if (groupId < VehicleTypeGroupId(_numOfVehicleTypeGroups)) {
		return (*_numOfVehicleTypes)[(short)groupId];
	}
	return Max_VehicleTypes + 1;
}


CU_HSTDEV bool CuVehicleTypeGroups::hasVehicleType(VehicleTypeGroupId groupId, VehicleTypeId typeId) const
{
	assert(groupId.isValid());
	assert(typeId.isValid());
	assert(groupId < VehicleTypeGroupId(_numOfVehicleTypeGroups));

	if (groupId < VehicleTypeGroupId(_numOfVehicleTypeGroups))
	{
		for (int j = 0; j < (*_numOfVehicleTypes)[(short)groupId]; j++)
		{
			if (_vehicleTypeIds->get((short)groupId, j) == typeId) return true;
		}
	}

	return false;
}


CU_HSTDEV VehicleTypeId CuVehicleTypeGroups::get(VehicleTypeGroupId groupId, int idx) const
{
	assert(groupId.isValid());
	assert(idx >= 0);
	assert(groupId < VehicleTypeGroupId(_numOfVehicleTypeGroups));
	assert(idx < (*_numOfVehicleTypes)[(short)groupId]);
	return _vehicleTypeIds->get((short)groupId, idx);
}
