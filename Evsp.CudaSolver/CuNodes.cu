
#include <assert.h>
#include "EvspLimits.h"
#include "CuNodes.h"




void CuNodes::init()
{
	for (int i = 0; i < Max_Nodes; i++) {
		_nodeType[i] = CuNodeType::INVALID;
	}
}


ushort CuNodes::getNumOfNodes()
{
	return _numOfNodes;
}


NodeId CuNodes::addNode(CuNodeType nodeType, short payloadId, VehicleTypeId vehTypeId)
{
	assert(_numOfNodes + 1 < Max_Nodes);
	assert(vehTypeId < VehicleTypeId(Max_VehicleTypes));
	ushort newId = _numOfNodes;
	_nodeType[newId] = nodeType;
	_payloadId[newId] = payloadId;
	_vehTypeId[newId] = vehTypeId;
	_numOfNodes++;
	return NodeId(newId);
}


CuNodeType CuNodes::getNodeType(NodeId nodeId)
{
	assert((short)nodeId < Max_Nodes);
	return _nodeType[(short)nodeId];
}


short CuNodes::getPayloadId(NodeId nodeId)
{
	assert((short)nodeId < Max_Nodes);
	return _payloadId[(short)nodeId];
}


CU_HSTDEV VehicleTypeId CuNodes::getVehTypeId(NodeId nodeId)
{
	assert((short)nodeId < Max_Nodes);
	return _vehTypeId[(short)nodeId];
}

