#include "CuPlans.h"

#include "cuda_runtime.h"
#include "CuPlan.h"
#include "CuConstructionGraph.h"


CuPlans::CuPlans(int population, int maxNumOfCirculations, int maxNumOfNodes)
	: _population(population), _maxNumOfCirculations(maxNumOfCirculations), _maxNumOfNodes(maxNumOfNodes), _deviceObject(0)
{
	assert(_population > 0);
	assert(_maxNumOfCirculations > 0);
	assert(_maxNumOfNodes > 0);
	
	_nodes = new Matrix3d<NodeId>(_maxNumOfNodes, _maxNumOfCirculations, _population);
	_edges = new Matrix3d<EdgeId>(_maxNumOfNodes, _maxNumOfCirculations, _population);

	_numOfCirculations = new CuVector1<int>(_population);

	_numOfNodes = new CuMatrix1<int>(_maxNumOfCirculations, _population);

	_startDepot = new CuMatrix1<StopId>(_maxNumOfCirculations, _population);
	_vehicleType = new CuMatrix1<VehicleTypeId>(_maxNumOfCirculations, _population);
}


CuPlans::CuPlans(const CuPlans &other)
{
	assert(false);
}


CuPlans::~CuPlans()
{
	if (_nodes) delete _nodes;
	if (_edges) delete _edges;
	if (_numOfCirculations) delete _numOfCirculations;
	if (_numOfNodes) delete _numOfNodes;
	if (_startDepot) delete _startDepot;
	if (_vehicleType) delete _vehicleType;

	if (_deviceObject) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _deviceObject));
	}
}


CU_HSTDEV void CuPlans::reInit(int solutionId)
{
	(*_numOfCirculations)[solutionId] = 0;
}


CuPlans* CuPlans::getDevPtr()
{
	if (!_deviceObject) {
		Matrix3d<NodeId> *tempNodes = _nodes;
		Matrix3d<EdgeId> *tempEdges = _edges;
		CuVector1<int> *tempCircNum = _numOfCirculations;
		CuMatrix1<int> *tempNodeNum= _numOfNodes;
		CuMatrix1<StopId> *tempStartDepot = _startDepot;
		CuMatrix1<VehicleTypeId> *tempVehicleType = _vehicleType;
		CuPlans* tempObj;

		_nodes = _nodes->getDevPtr();
		_edges = _edges->getDevPtr();
		_numOfCirculations = _numOfCirculations->getDevPtr();
		_numOfNodes = _numOfNodes->getDevPtr();
		_startDepot = _startDepot->getDevPtr();
		_vehicleType = _vehicleType->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempObj, sizeof(CuPlans)));
		CUDA_CHECK(cudaMemcpy(tempObj, this, sizeof(CuPlans), cudaMemcpyHostToDevice));
		_deviceObject = tempObj;

		_nodes = tempNodes;
		_edges = tempEdges;
		_numOfCirculations = tempCircNum;
		_numOfNodes = tempNodeNum;
		_startDepot = tempStartDepot;
		_vehicleType = tempVehicleType;
	}
	return _deviceObject;
}


void CuPlans::copyToDevice()
{
	if (_deviceObject) {
	
		_nodes->copyToDevice();
		_edges->copyToDevice();
		_numOfCirculations->copyToDevice();
		_numOfNodes->copyToDevice();
		_startDepot->copyToDevice();
		_vehicleType->copyToDevice();

		Matrix3d<NodeId> *tempNodes = _nodes;
		Matrix3d<EdgeId> *tempEdges = _edges;
		CuVector1<int> *tempCircNum = _numOfCirculations;
		CuMatrix1<int> *tempNodeNum = _numOfNodes;
		CuMatrix1<StopId> *tempStartDepot = _startDepot;
		CuMatrix1<VehicleTypeId> *tempVehicleType = _vehicleType;
		CuPlans* tempObj = _deviceObject;

		_nodes = _nodes->getDevPtr();
		_edges = _edges->getDevPtr();
		_numOfCirculations = _numOfCirculations->getDevPtr();
		_numOfNodes = _numOfNodes->getDevPtr();
		_startDepot = _startDepot->getDevPtr();
		_vehicleType = _vehicleType->getDevPtr();
		_deviceObject = 0;

		CUDA_CHECK(cudaMemcpy(tempObj, this, sizeof(CuPlans), cudaMemcpyHostToDevice));

		_deviceObject = tempObj;

		_nodes = tempNodes;
		_edges = tempEdges;
		_numOfCirculations = tempCircNum;
		_numOfNodes = tempNodeNum;
		_startDepot = tempStartDepot;
		_vehicleType = tempVehicleType;
	}
}


void CuPlans::copyToHost()
{
	if (_deviceObject) {
		Matrix3d<NodeId> *tempNodes = _nodes;
		Matrix3d<EdgeId> *tempEdges = _edges;
		CuVector1<int> *tempCircNum = _numOfCirculations;
		CuMatrix1<int> *tempNodeNum = _numOfNodes;
		CuMatrix1<StopId> *tempStartDepot = _startDepot;
		CuMatrix1<VehicleTypeId> *tempVehicleType = _vehicleType;
		CuPlans* tempObj = _deviceObject;

		CUDA_CHECK(cudaMemcpy(this, _deviceObject, sizeof(CuPlans), cudaMemcpyDeviceToHost));

		_nodes = tempNodes;
		_edges = tempEdges;
		_numOfCirculations = tempCircNum;
		_numOfNodes = tempNodeNum;
		_startDepot = tempStartDepot;
		_vehicleType = tempVehicleType;

		_nodes->copyToHost();
		_edges->copyToHost();
		_numOfCirculations->copyToHost();
		_numOfNodes->copyToHost();
		_startDepot->copyToHost();
		_vehicleType->copyToHost();

		_deviceObject = tempObj;
	}
}


CU_HSTDEV CirculationId CuPlans::addNewCirculation(int solutionId, StopId startDepot, VehicleTypeId vehicleType)
{
	assert(solutionId < _population);
	CirculationId retVal;
	int &currentNumOfCirculations = (*_numOfCirculations)[solutionId];
	assert(currentNumOfCirculations < _maxNumOfCirculations);
	if (currentNumOfCirculations < _maxNumOfCirculations) {
		retVal = CirculationId(currentNumOfCirculations);
		_numOfNodes->set(currentNumOfCirculations, solutionId, 0);
		_startDepot->set(currentNumOfCirculations, solutionId, startDepot);
		_vehicleType->set(currentNumOfCirculations, solutionId, vehicleType);
		currentNumOfCirculations++;
	}
	else {
		// TODO Lösung als ungültig markieren!
		printf("Maximalanzahl der Umläufe überschritten!\n");
	}
	return retVal;
}


CU_HSTDEV CircStepIndex CuPlans::appendNode(int solutionId, CirculationId circulationId, NodeId newNode, EdgeId selectedEdge)
{
	assert(solutionId >= 0);
	assert(circulationId >= 0);
	assert(solutionId < _population);
	assert(newNode.isValid());
	//assert(selectedEdge.isValid());

	int currentNumOfCirculations = (*_numOfCirculations)[solutionId];
	assert(circulationId < currentNumOfCirculations);

	int &currentNumOfNodes = _numOfNodes->itemAt((short)circulationId, solutionId);
	assert(currentNumOfNodes < _maxNumOfNodes);

	if (currentNumOfNodes < _maxNumOfNodes) {
		CircStepIndex retVal(currentNumOfNodes);
		_nodes->set((short)retVal, (short)circulationId, solutionId, newNode);
		_edges->set((short)retVal, (short)circulationId, solutionId, selectedEdge);
		currentNumOfNodes++;
		return retVal;
	}

	printf("Fehler: Die maximale Länge eines Umlaufs wurde überschritten (solutionId=%i, numOfNodes=%i)!\n",
		solutionId, currentNumOfNodes);

	return CircStepIndex::invalid();
}


//CU_HSTDEV void CuPlans::revertLastDecision(int solutionId, CirculationId circulationId)
//{
//	assert(solutionId >= 0);
//	assert(solutionId < _population);
//	assert(circulationId < (*_numOfCirculations)[solutionId]);
//
//	if (circulationId < 0) return;
//
//	int &currentNumOfNodes = _numOfNodes->itemAt((short)circulationId, solutionId);
//
//	if (currentNumOfNodes > 0) {
//		currentNumOfNodes--;
//	}
//}


CU_HSTDEV NodeId CuPlans::getNodeId(int solutionId, CirculationId circulationId, CircStepIndex stepIndex)
{
	NodeId retVal;

	assert(solutionId >= 0);
	assert(solutionId < _population);
	if (solutionId >= 0 && solutionId < _population) {
		assert(circulationId >= 0);
		assert(circulationId < getNumOfCirculations(solutionId));
		if (circulationId >= 0 && circulationId < getNumOfCirculations(solutionId)) {
			assert(stepIndex >= 0);
			assert(stepIndex < getNumOfNodes(solutionId, circulationId));
			if (stepIndex >= 0 && stepIndex < getNumOfNodes(solutionId, circulationId)) {
				retVal = _nodes->get((short)stepIndex, (short)circulationId, solutionId);
			}
			else {
				printf("CuPlans::getNodeId(): stepIndex out of bounds (%i)!", (short)stepIndex);
			}
		}
		else {
			printf("CuPlans::getNodeId(): circulationId out of bounds (%i)!", (short)circulationId);
		}
	}
	else {
		printf("CuPlans::getNodeId(): solutionId out of bounds (%i)!", solutionId);
	}

	return retVal;
}


CU_HSTDEV EdgeId CuPlans::getEdgeId(int solutionId, CirculationId circulationId, CircStepIndex stepIndex)
{
	assert(solutionId >= 0);
	assert(circulationId >= 0);
	assert(stepIndex >= 0);
	assert(solutionId < _population);
	assert(circulationId < getNumOfCirculations(solutionId));
	assert(stepIndex < getNumOfNodes(solutionId, circulationId));
	return _edges->get((short)stepIndex, (short)circulationId, solutionId);
}


CU_HSTDEV int CuPlans::getNumOfCirculations(int solutionId) const
{
	assert(solutionId >= 0);
	assert(solutionId < _population);
	return (*_numOfCirculations)[solutionId];
}


CU_HSTDEV int CuPlans::getNumOfNodes(int solutionId, CirculationId circulationId) const
{
	assert(solutionId >= 0);
	assert(circulationId >= 0);
	assert(solutionId < _population);
	assert(circulationId < (*_numOfCirculations)[solutionId]);
	int retVal = _numOfNodes->get((short)circulationId, solutionId);
	return retVal;
}


CU_HSTDEV StopId CuPlans::getDepotId(int solutionId, CirculationId circulationId) const
{
	assert(solutionId >= 0);
	assert(circulationId >= 0);
	assert(solutionId < _population);
	assert(circulationId < (*_numOfCirculations)[solutionId]);
	return _startDepot->get((short)circulationId, solutionId);
}

CU_HSTDEV void CuPlans::setDepotId(int solutionId, CirculationId circulationId, StopId depotId)
{
	assert(solutionId >= 0);
	assert(circulationId >= 0);
	assert(solutionId < _population);
	assert(circulationId < (*_numOfCirculations)[solutionId]);
	_startDepot->set((short)circulationId, solutionId, depotId);
}

CU_HSTDEV VehicleTypeId CuPlans::getVehicleTypeId(int solutionId, CirculationId circulationId) const
{
	assert(solutionId >= 0);
	assert(circulationId >= 0);
	assert(solutionId < _population);
	assert(circulationId < (*_numOfCirculations)[solutionId]);
	return _vehicleType->get((short)circulationId, solutionId);
}

CU_HSTDEV void CuPlans::setVehicleTypeId(int solutionId, CirculationId circulationId, VehicleTypeId vehicleTypeId)
{
	assert(solutionId >= 0);
	assert(circulationId >= 0);
	assert(solutionId < _population);
	assert(circulationId < (*_numOfCirculations)[solutionId]);
	_vehicleType->set((short)circulationId, solutionId, vehicleTypeId);
}


std::shared_ptr<CuPlan> CuPlans::getPlan(int solutionId)
{
	assert(solutionId >= 0);
	assert(solutionId < _population);
	int numOfCirculations = (*_numOfCirculations)[solutionId];

	shared_ptr<CuPlan> retVal = make_shared<CuPlan>(_maxNumOfCirculations, _maxNumOfNodes);

	for (CirculationId i(0); i < numOfCirculations; i++) {
		int numOfNodes = _numOfNodes->get((short)i, solutionId);
		retVal->addNewCirculation(_startDepot->get((short)i, solutionId), _vehicleType->get((short)i, solutionId));
		for (CircStepIndex j(0); j < numOfNodes; j++) {
			NodeId node = getNodeId(solutionId, i, j);
			EdgeId edge = getEdgeId(solutionId, i, j);
			retVal->appendNode(i, node, edge);
		}
	}

	return retVal;
}


CU_HSTDEV CuPlan& CuPlans::getPlan(int solutionId, CuPlan &result)
{
	assert(solutionId >= 0);
	assert(solutionId < _population);
	
	result.reInit();

	int numOfCirculations = (*_numOfCirculations)[solutionId];

	for (CirculationId i(0); i < numOfCirculations; i++) {
		int numOfNodes = _numOfNodes->get((short)i, solutionId);
		result.addNewCirculation(_startDepot->get((short)i, solutionId), _vehicleType->get((short)i, solutionId));
		for (CircStepIndex j(0); j < numOfNodes; j++) {
			NodeId node = getNodeId(solutionId, i, j);
			EdgeId edge = getEdgeId(solutionId, i, j);
			result.appendNode(i, node, edge);
		}
	}

	return result;
}


CU_HSTDEV void CuPlans::dump(CuConstructionGraph *ptn, int solutionId)
{
	for (CirculationId i(0); i < getNumOfCirculations(solutionId); i++) {
		for (CircStepIndex j(0); j < getNumOfNodes(solutionId, i); j++) {
			NodeId cn = getNodeId(solutionId, i, j);
			EdgeId se = getEdgeId(solutionId, i, j);

			CuNodeType nodeType = ptn->nodes.getNodeType(cn);
			printf("* %i", (short)cn);
			if (se.isValid()) {
				printf(" --(%.1f)--> ", ptn->edges->getWeight(cn, se));
			}
		}
		printf("*\n");
	}
}

void CuPlans::dump(shared_ptr<CuConstructionGraph> ptn, int solutionId)
{
	assert(solutionId >= 0);
	assert(solutionId < _population);

	dump(ptn.get(), solutionId);
}
