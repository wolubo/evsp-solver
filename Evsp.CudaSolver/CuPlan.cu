#include "CuPlan.h"

#include "cuda_runtime.h"
#include "CuConstructionGraph.h"


CuPlan::CuPlan(int maxNumOfCirculations, int maxNumOfNodes)
	: _maxNumOfCirculations(maxNumOfCirculations), _maxNumOfNodes(maxNumOfNodes), _numOfCirculations(0),  _deviceObject(0)
{
	assert(_maxNumOfCirculations > 0);
	assert(_maxNumOfNodes > 0);

	_nodes = new CuMatrix1<NodeId>(_maxNumOfNodes, _maxNumOfCirculations);
	_edges = new CuMatrix1<EdgeId>(_maxNumOfNodes, _maxNumOfCirculations);

	_numOfNodes = new CuVector1<int>(_maxNumOfCirculations);
	_startDepot = new CuVector1<StopId>(_maxNumOfCirculations);
	_vehicleType = new CuVector1<VehicleTypeId>(_maxNumOfCirculations);
}


CuPlan::CuPlan(const CuPlan &other)
{
	assert(false);
}


CuPlan::~CuPlan()
{
	if (_nodes) delete _nodes;
	if (_edges) delete _edges;
	if (_numOfNodes) delete _numOfNodes;
	if (_startDepot) delete _startDepot;
	if (_vehicleType) delete _vehicleType;
	if (_deviceObject) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _deviceObject));
	}
}


CU_HSTDEV void CuPlan::reInit()
{
	_numOfCirculations = 0;
}


CuPlan* CuPlan::getDevPtr()
{
	if (!_deviceObject) {
		CuMatrix1<NodeId> *tempNodes = _nodes;
		CuMatrix1<EdgeId> *tempEdges = _edges;
		CuVector1<int> *tempNodeNum = _numOfNodes;
		CuVector1<StopId> *tempStartDepot = _startDepot;
		CuVector1<VehicleTypeId> *tempVehicleType = _vehicleType;
		CuPlan* tempObj;

		_nodes = _nodes->getDevPtr();
		_edges = _edges->getDevPtr();
		_numOfNodes = _numOfNodes->getDevPtr();
		_startDepot= _startDepot->getDevPtr();
		_vehicleType = _vehicleType->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempObj, sizeof(CuPlan)));
		CUDA_CHECK(cudaMemcpy(tempObj, this, sizeof(CuPlan), cudaMemcpyHostToDevice));
		_deviceObject = tempObj;

		_nodes = tempNodes;
		_edges = tempEdges;
		_numOfNodes = tempNodeNum;
		_startDepot = tempStartDepot;
		_vehicleType = tempVehicleType;
	}
	return _deviceObject;
}


void CuPlan::copyToDevice()
{
	if (_deviceObject) {

		_nodes->copyToDevice();
		_edges->copyToDevice();
		_numOfNodes->copyToDevice();
		_startDepot->copyToDevice();
		_vehicleType->copyToDevice();

		CuMatrix1<NodeId> *tempNodes = _nodes;
		CuMatrix1<EdgeId> *tempEdges = _edges;
		CuVector1<int> *tempNodeNum = _numOfNodes;
		CuVector1<StopId> *tempStartDepot = _startDepot;
		CuVector1<VehicleTypeId> *tempVehicleType = _vehicleType;
		CuPlan* tempObj = _deviceObject;

		_nodes = _nodes->getDevPtr();
		_edges = _edges->getDevPtr();
		_numOfNodes = _numOfNodes->getDevPtr();
		_startDepot = _startDepot->getDevPtr();
		_vehicleType = _vehicleType->getDevPtr();
		_deviceObject = 0;

		CUDA_CHECK(cudaMemcpy(tempObj, this, sizeof(CuPlan), cudaMemcpyHostToDevice));

		_deviceObject = tempObj;

		_nodes = tempNodes;
		_edges = tempEdges;
		_numOfNodes = tempNodeNum;
		_startDepot = tempStartDepot;
		_vehicleType = tempVehicleType;

	}
}


void CuPlan::copyToHost()
{
	if (_deviceObject) {
		CuMatrix1<NodeId> *tempNodes = _nodes;
		CuMatrix1<EdgeId> *tempEdges = _edges;
		CuVector1<int> *tempNodeNum = _numOfNodes;
		CuVector1<StopId> *tempStartDepot = _startDepot;
		CuVector1<VehicleTypeId> *tempVehicleType = _vehicleType;
		CuPlan* tempObj = _deviceObject;

		CUDA_CHECK(cudaMemcpy(this, _deviceObject, sizeof(CuPlan), cudaMemcpyDeviceToHost));

		_nodes = tempNodes;
		_edges = tempEdges;
		_numOfNodes = tempNodeNum;
		_startDepot = tempStartDepot;
		_vehicleType = tempVehicleType;

		_nodes->copyToHost();
		_edges->copyToHost();
		_numOfNodes->copyToHost();
		_startDepot->copyToHost();
		_vehicleType->copyToHost();

		_deviceObject = tempObj;
	}
}


CU_HSTDEV CirculationId CuPlan::addNewCirculation(StopId startDepot, VehicleTypeId vehicleType)
{
	CirculationId retVal;
	//assert(_numOfCirculations < _maxNumOfCirculations);
	if (_numOfCirculations < _maxNumOfCirculations) {
		retVal = CirculationId(_numOfCirculations);
		(*_numOfNodes)[_numOfCirculations] = 0;
		(*_startDepot)[_numOfCirculations] = startDepot;
		(*_vehicleType)[_numOfCirculations] = vehicleType;
		_numOfCirculations++;
	}
	else {
		// TODO Lösung als ungültig markieren!
		printf("Maximalanzahl der Umläufe überschritten!\n");
		assert(false);
	}
	return retVal;
}


CU_HSTDEV CircStepIndex CuPlan::appendNode(CirculationId circulationId, NodeId newNode, EdgeId selectedEdge)
{
	assert(circulationId >= 0);
	assert(newNode.isValid());
	assert(circulationId < _numOfCirculations);

	int &currentNumOfNodes = (*_numOfNodes)[(short)circulationId];
	assert(currentNumOfNodes < _maxNumOfNodes);

	if (currentNumOfNodes < _maxNumOfNodes) {
		CircStepIndex retVal(currentNumOfNodes);
		_nodes->set((short)retVal, (short)circulationId, newNode);
		_edges->set((short)retVal, (short)circulationId, selectedEdge);
		currentNumOfNodes++;
		return retVal;
	}

	printf("Fehler: Die maximale Anzahl von Aktionen eines Umlaufs wurde überschritten (maxNumOfNodes=%i)!\n",
		currentNumOfNodes);

	return CircStepIndex::invalid();
}


//CU_HSTDEV void CuPlan::revertLastDecision(CirculationId circulationId)
//{
//	assert(circulationId < _numOfCirculations);
//
//	if (circulationId < 0) return;
//
//	int &currentNumOfNodes = (*_numOfNodes)[(short)circulationId];
//
//	if (currentNumOfNodes > 0) {
//		currentNumOfNodes--;
//	}
//}


CU_HSTDEV NodeId CuPlan::getNodeId(CirculationId circulationId, CircStepIndex stepIndex)
{
	assert(circulationId >= 0);
	assert(stepIndex >= 0);
	assert(circulationId < _numOfCirculations);
	assert(stepIndex < getNumOfNodes(circulationId));
	return _nodes->get((short)stepIndex, (short)circulationId);
}


CU_HSTDEV EdgeId CuPlan::getEdgeId(CirculationId circulationId, CircStepIndex stepIndex)
{
	assert(circulationId >= 0);
	assert(stepIndex >= 0);
	assert(circulationId < _numOfCirculations);
	assert(stepIndex < getNumOfNodes(circulationId));
	return _edges->get((short)stepIndex, (short)circulationId);
}


CU_HSTDEV int CuPlan::getNumOfNodes(CirculationId circulationId)
{
	assert(circulationId >= 0);
	assert(circulationId < _numOfCirculations);
	int retVal = (*_numOfNodes)[(short)circulationId];
	return retVal;
}


void CuPlan::dump(shared_ptr<CuConstructionGraph> ptn)
{
	ptn->copyToHost();

	bool first = true;

	for (CirculationId i(0); i < getNumOfCirculations(); i++) {
		for (CircStepIndex j(0); j < getNumOfNodes(i); j++) {
			NodeId cn = getNodeId(i, j);
			EdgeId se = getEdgeId(i, j);

			CuNodeType nodeType = ptn->nodes.getNodeType(cn);
			if (nodeType == CuNodeType::RootNode) {
				if (first) {
					first = false;
				}
				else {
					cout << "*";
					cout << endl;
				}
				cout << "* " << (short)cn;
				assert(se.isValid());
				cout << " --(";
				cout << ptn->edges->getWeight(cn, se);
				cout << ")--> ";
			}
			else {
				cout << (short)cn;
				if (se.isValid()) {
					cout << " --(";
					cout << ptn->edges->getWeight(cn, se);
					cout << ")--> ";
				}
			}
		}
		cout << endl;
	}
}
