#include "CuEdges.h"

#include <assert.h>
#include <stdio.h>
#include <cstring>
#include "cuda_runtime.h"



CuEdges::CuEdges(int numOfNodes, int maxNumOfOutEdges)
	: _numOfNodes(numOfNodes), _maxNumOfOutEdges(maxNumOfOutEdges), _devicePtr(0)
{
	assert(numOfNodes < Max_Nodes);
	assert(maxNumOfOutEdges < Max_EdgesPerNode);

	_weight = new CuMatrix1<float>(numOfNodes, maxNumOfOutEdges);
	_visitCounter = new CuMatrix1<int>(numOfNodes, maxNumOfOutEdges);
	_numOfOutEdges = new CuVector1<ushort>(numOfNodes);
	_trgNodeId = new CuMatrix1<NodeId>(numOfNodes, maxNumOfOutEdges);
}


CuEdges::~CuEdges()
{
	if (_weight) delete _weight;
	if (_visitCounter) delete _visitCounter;
	if (_numOfOutEdges) delete _numOfOutEdges;
	if (_trgNodeId) delete _trgNodeId;
	if (_devicePtr) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}


CuEdges* CuEdges::getDevPtr()
{
	if (!_devicePtr) {
		CuMatrix1<float> *tempWeight = _weight;
		CuMatrix1<int> *tempCounter = _visitCounter;
		CuVector1<ushort> *tempEdges = _numOfOutEdges;
		CuMatrix1<NodeId> *tempTrgNode = _trgNodeId;
		CuEdges *tempDevPtr;

		_weight = _weight->getDevPtr();
		_visitCounter = _visitCounter->getDevPtr();
		_numOfOutEdges = _numOfOutEdges->getDevPtr();
		_trgNodeId = _trgNodeId->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempDevPtr, sizeof(CuEdges)));
		CUDA_CHECK(cudaMemcpy(tempDevPtr, this, sizeof(CuEdges), cudaMemcpyHostToDevice));

		_devicePtr = tempDevPtr;
		_weight = tempWeight;
		_visitCounter = tempCounter;
		_numOfOutEdges = tempEdges;
		_trgNodeId = tempTrgNode;
	}
	return _devicePtr;
}


void CuEdges::copyToHost()
{
	if (_devicePtr) {
		CuMatrix1<float> *tempWeight = _weight;
		CuMatrix1<int> *tempCounter = _visitCounter;
		CuVector1<ushort> *tempEdges = _numOfOutEdges;
		CuMatrix1<NodeId> *tempTrgNode = _trgNodeId;
		CuEdges* tempObj = _devicePtr;

		CUDA_CHECK(cudaMemcpy(this, _devicePtr, sizeof(CuEdges), cudaMemcpyDeviceToHost));

		_weight = tempWeight;
		_visitCounter = tempCounter;
		_numOfOutEdges = tempEdges;
		_trgNodeId = tempTrgNode;

		_weight->copyToHost();
		_visitCounter->copyToHost();
		_numOfOutEdges->copyToHost();
		_trgNodeId->copyToHost();

		_devicePtr = tempObj;
	}
}


void CuEdges::copyToDevice()
{
	if (_devicePtr) {
		_weight->copyToDevice();
		_visitCounter->copyToDevice();
		_numOfOutEdges->copyToDevice();
		_trgNodeId->copyToDevice();
	}
}


CU_HSTDEV EdgeId CuEdges::addEdge(NodeId fromNode, NodeId toNode, float initWeight)
{
	assert((short)fromNode < _numOfNodes);
	assert(initWeight >= 0.0);

	ushort newId = 0;
	ushort &value = (*_numOfOutEdges)[(short)fromNode];
	newId = value;
	value++;
	assert(value < _maxNumOfOutEdges);
	assert(value < Max_EdgesPerNode);
	assert(value == newId + 1);
	assert((*_numOfOutEdges)[(short)fromNode] == value);
	assert((*_numOfOutEdges)[(short)fromNode] == (newId + 1));

	_trgNodeId->set((short)fromNode, newId, toNode);
	_weight->set((short)fromNode, newId, initWeight);
	_visitCounter->set((short)fromNode, newId, 0);

	return EdgeId(newId);
}


float CuEdges::getWeight(NodeId fromNode, EdgeId edgeIndex)
{
//#pragma warning(suppress : 4018)
	assert((short)fromNode < _numOfNodes);
	assert(edgeIndex < (*_numOfOutEdges)[(short)fromNode]);
	return _weight->get((short)fromNode, (short)edgeIndex);
}


int CuEdges::getVisitCounter(NodeId fromNode, EdgeId edgeIndex)
{
	assert((short)fromNode < _numOfNodes);
	assert(edgeIndex < (*_numOfOutEdges)[(short)fromNode]);
	return _visitCounter->get((short)fromNode, (short)edgeIndex);
}


void CuEdges::addToVisitCounter(NodeId fromNode, EdgeId edgeIndex, int value)
{
	assert((short)fromNode < _numOfNodes);
	assert(edgeIndex < (*_numOfOutEdges)[(short)fromNode]);
	_visitCounter += value;
}

void CuEdges::resetVisitCounter(NodeId fromNode, EdgeId edgeIndex)
{
	assert((short)fromNode < _numOfNodes);
	assert(edgeIndex < (*_numOfOutEdges)[(short)fromNode]);
	_visitCounter = 0;
}

NodeId CuEdges::getTargetNode(NodeId fromNode, EdgeId edgeIndex)
{
	assert((short)fromNode < _numOfNodes);
	assert(edgeIndex < (*_numOfOutEdges)[(short)fromNode]);
	return _trgNodeId->get((short)fromNode, (short)edgeIndex);
}


int CuEdges::getNumOfOutEdges(NodeId fromNode)
{
	assert((short)fromNode < _numOfNodes);
	return (*_numOfOutEdges)[(short)fromNode];
}


void CuEdges::setWeight(NodeId fromNode, EdgeId edgeIndex, float newWeight)
{
	assert((short)fromNode < _numOfNodes);
	assert(edgeIndex < (*_numOfOutEdges)[(short)fromNode]);
	if (newWeight < 0.0f) {
		printf("CuEdges::setWeight(): newWeight < 0! (fromNode=%i, edgeIndex=%i, newWeight=%f)\n", (short)fromNode, (short)edgeIndex, newWeight);
		assert(false);
	}
	_weight->set((short)fromNode, (short)edgeIndex, newWeight);
}


CU_HSTDEV void CuEdges::addWeight(NodeId fromNode, EdgeId edgeIndex, float newTrack)
{
	assert((short)fromNode < _numOfNodes);
	assert(edgeIndex < (*_numOfOutEdges)[(short)fromNode]);
	float oldTrack = _weight->get((short)fromNode, (short)edgeIndex);
	_weight->set((short)fromNode, (short)edgeIndex, oldTrack + newTrack);
}


CU_HSTDEV void CuEdges::fadeTracks(float fading)
{
	float fadeFactor = (1.0f - fading);
	for (NodeId n(0); n < _numOfNodes; n++) {
		for (EdgeId e(0); e < getNumOfOutEdges(n); e++) {
			float oldTrack = _weight->get((short)n, (short)e);
			_weight->set((short)n, (short)e, oldTrack * fadeFactor);
		}
	}
}
