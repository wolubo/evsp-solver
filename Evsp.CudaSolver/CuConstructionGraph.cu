#include "CuConstructionGraph.h"

#include <iostream>
#include <iomanip>
#include "cuda_runtime.h"

#include "EVSP.BaseClasses/Typedefs.h"
#include "CuBoolVector1.hpp"
#include "PtnKernels.h"
#include "PtnCreateEdgesCpu.h"
#include "EVSP.BaseClasses/Stopwatch.h"
#include "CudaCheck.h"



CuConstructionGraph::CuConstructionGraph(std::shared_ptr<CuProblem> problem, float initialWeight, PlattformConfig plattform, bool performCheck)
	: nodes(), _deviceObject(0)
{
	Stopwatch stopwatch;
	stopwatch.start();

	nodes.init();
	int numOfServiceTrips = problem->getServiceTrips().getNumOfServiceTrips();
	int numOfChargingStations = problem->getStops().getNumOfChargingStations();

	createNodes(problem);

	edges = new CuEdges(nodes.getNumOfNodes(), numOfServiceTrips + numOfChargingStations + 1);
	createEdges(problem, initialWeight, plattform);

	if(performCheck) check(problem);

	stopwatch.stop("ACO-Entscheidungsnetz erzeugt (GPU): ");
}


CuConstructionGraph::~CuConstructionGraph()
{
	if (edges) delete edges;
	if (_deviceObject)
	{
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _deviceObject));
	}
}


CuConstructionGraph* CuConstructionGraph::getDevPtr()
{
	if (!_deviceObject)
	{
		CuConstructionGraph *tempDevObj;
		CuEdges *tempEdges = edges;

		edges = edges->getDevPtr();

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&tempDevObj, sizeof(CuConstructionGraph)));
		
		CUDA_CHECK(cudaMemcpy(tempDevObj, this, sizeof(CuConstructionGraph), cudaMemcpyHostToDevice));

		edges = tempEdges;

		_deviceObject = tempDevObj;
	}
	return _deviceObject;
}


void CuConstructionGraph::copyToHost()
{
	if (_deviceObject) {
		CuEdges *tempEdges = edges;
		CuConstructionGraph* tempDevObj = _deviceObject;
		CUDA_CHECK(cudaMemcpy(this, tempDevObj, sizeof(CuConstructionGraph), cudaMemcpyDeviceToHost));
		edges = tempEdges;
		edges->copyToHost();
		_deviceObject = tempDevObj;
	}
}


void CuConstructionGraph::createNodes(std::shared_ptr<CuProblem> problem)
{
	// Erzeuge den Wurzelknoten.
	nodes.addNode(CuNodeType::RootNode, -1, VehicleTypeId::invalid()); // Der Wurzelknoten hat immer die Id '0', keine Payload und keine Fahrzeugtypen. 

	// Erzeuge die Knoten für alle Kombinationen aus Depots und Fahrzeugtypen.
	for (DepotId i(0); i <= problem->getStops().lastDepotId(); i++) {
		StopId depotId = problem->getStops().getStopIdOfDepot(i);
		for (VehicleTypeId vehTypeId(0); vehTypeId < VehicleTypeId(problem->getVehicleTypes().getNumOfVehicleTypes()); vehTypeId++) {
			nodes.addNode(CuNodeType::VehTypeDepotNode, (short)depotId, vehTypeId);
		}
	}

	// Erzeuge die Knoten für die Ladestationen. 
	for (ChargingStationId i(0); i <= problem->getStops().lastChargingStationId(); i++) {
		StopId chargingStationId = problem->getStops().getStopIdOfChargingStation(i);
		nodes.addNode(CuNodeType::ChargingStationNode, (short)chargingStationId, VehicleTypeId::invalid());
	}

	// Erzeuge die Knoten für die Servicefahrten.
	for (ushort i = 0; i < problem->getServiceTrips().getNumOfServiceTrips(); i++) {
		nodes.addNode(CuNodeType::ServiceTripNode, i, VehicleTypeId::invalid());
	}
}


void CuConstructionGraph::createEdges(std::shared_ptr<CuProblem> problem, float initialWeight, PlattformConfig plattform)
{
	shared_ptr<VehicleTypeGroupIntersection> vtgIntersect = MatrixCreator::createVehicleTypeGroupIntersection(problem);
	int numberOfNodes = nodes.getNumOfNodes();

	if (plattform == GPU) {
		CuLockVector1 *lock = new CuLockVector1(numberOfNodes);

		int blockSize = 32; // Threads pro Block
		int minGridSize;
		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)createPtnEdgesKernel));
		blockSize = (int)sqrt(blockSize);
		int numOfBlocks = (nodes.getNumOfNodes() + blockSize - 1) / blockSize;
		dim3 dimGrid(numOfBlocks, numOfBlocks);
		dim3 dimBlock(blockSize, blockSize);

		createPtnEdgesKernel << <dimGrid, dimBlock >> > (getDevPtr(), lock->getDevPtr(), problem->getDevPtr(), vtgIntersect->getDevPtr(), initialWeight, nodes.getNumOfNodes());
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		delete lock;
	}
	else {
		createPtnEdgesOnCpu(this, problem, vtgIntersect, initialWeight, numberOfNodes);
		copyToHost();
	}
}


string nodeTypeToString(CuNodeType nodeType)
{
	string retVal;
	switch (nodeType) {
	case CuNodeType::RootNode:
		retVal = "RootNode";
		break;
	case CuNodeType::VehTypeDepotNode:
		retVal = "VehTypeDepotNode";
		break;
	case CuNodeType::ServiceTripNode:
		retVal = "ServiceTripNode";
		break;
	case CuNodeType::ChargingStationNode:
		retVal = "ChargingStationNode";
		break;
	default:
		retVal = "<unknown>";
	}
	return retVal;
}


void CuConstructionGraph::printStatistic()
{
	copyToHost();

	const int numOfNodeTypes = 5; // Anzahl der Literale im Enum CuNodeType.

	cout << endl << "Decision-Net-Statistic:" << endl;
	cout << "Total #Nodes: " << nodes.getNumOfNodes() << endl;

	ushort numOfRootNodes = 0;
	ushort numOfVehTypeDepotNodes = 0;
	ushort numOfServiceTripNodes = 0;
	ushort numOfChargingStationNodes = 0;
	int totalNumOfEdges = 0;
	ushort minNumOfOutEdges = Max_EdgesPerNode;
	CuNodeType minOutType;
	ushort maxNumOfOutEdges = 0;
	CuNodeType maxOutType;
	int numOfEdgesByType[numOfNodeTypes][numOfNodeTypes] = { {0,0,0,0,0},{ 0,0,0,0,0},{ 0,0,0,0,0},{ 0,0,0,0,0},{0,0,0,0,0} };
	ushort numOfInEdges[Max_Nodes];
	for (ushort i = 0; i < nodes.getNumOfNodes(); i++) numOfInEdges[i] = 0;
	for (NodeId i = NodeId(0); i < NodeId(nodes.getNumOfNodes()); i++) {
		CuNodeType sourceNodeType = nodes.getNodeType(i);
		switch (sourceNodeType) {
		case CuNodeType::RootNode: numOfRootNodes++;
			break;
		case CuNodeType::VehTypeDepotNode: numOfVehTypeDepotNodes++;
			break;
		case CuNodeType::ServiceTripNode: numOfServiceTripNodes++;
			break;
		case CuNodeType::ChargingStationNode: numOfChargingStationNodes++;
			break;
		}

		ushort numOfEdges = edges->getNumOfOutEdges(i);
		totalNumOfEdges += numOfEdges;
		if (numOfEdges < minNumOfOutEdges) {
			minNumOfOutEdges = numOfEdges;
			minOutType = sourceNodeType;
		}
		if (numOfEdges > maxNumOfOutEdges) {
			maxNumOfOutEdges = numOfEdges;
			maxOutType = sourceNodeType;
		}

		for (EdgeId edgeId = EdgeId(0); edgeId < EdgeId(numOfEdges); edgeId++) {
			NodeId targetNodeId = edges->getTargetNode(i, EdgeId(edgeId));
			CuNodeType targetNodeType = nodes.getNodeType(targetNodeId);
			assert(sourceNodeType < numOfNodeTypes);
			assert(targetNodeType < numOfNodeTypes);
			numOfEdgesByType[sourceNodeType][targetNodeType]++;
			numOfInEdges[(short)targetNodeId]++; // Eingehende Kanten zählen.
		}
	}

	ushort minNumOfInEdges = Max_EdgesPerNode;
	ushort minInId = 0;
	CuNodeType minInType;
	ushort maxNumOfInEdges = 0;
	CuNodeType maxInType;
	for (ushort i = 1; i < nodes.getNumOfNodes(); i++) { // Ab 1, da die Root-Node hier nicht interessiert.
		if (numOfInEdges[i] > maxNumOfInEdges) {
			maxNumOfInEdges = numOfInEdges[i];
			maxInType = nodes.getNodeType(NodeId(i));
		}
		if (numOfInEdges[i] < minNumOfInEdges) {
			minNumOfInEdges = numOfInEdges[i];
			minInType = nodes.getNodeType(NodeId(i));
			minInId = i;
		}
	}

	cout << "   #RootNodes:            " << numOfRootNodes << endl;
	cout << "   #VehTypeDepotNodes:    " << numOfVehTypeDepotNodes << endl;
	cout << "   #ServiceTripNodes:     " << numOfServiceTripNodes << endl;
	cout << "   #ChargingStationNodes: " << numOfChargingStationNodes << endl;

	cout << "Total #Edges: " << totalNumOfEdges << endl;
	cout << "   avg. # out-edges: " << fixed << setprecision(1) << totalNumOfEdges / (float)nodes.getNumOfNodes() << endl;
	cout << "   min. # out-edges: " << minNumOfOutEdges << " (" << nodeTypeToString(minOutType) << ")" << endl;
	cout << "   max. # out-edges: " << maxNumOfOutEdges << " (" << nodeTypeToString(maxOutType) << ")" << endl;
	cout << "   min. # in-edges: " << minNumOfInEdges << " (" << nodeTypeToString(minInType) << ", id=" << minInId << ")" << endl;
	cout << "   max. # in-edges: " << maxNumOfInEdges << " (" << nodeTypeToString(maxInType) << ")" << endl;

	for (ushort i = 0; i < numOfNodeTypes; i++) {
		for (ushort j = 0; j < numOfNodeTypes; j++) {
			int numOfEdges = numOfEdgesByType[i][j];
			if (numOfEdges > 0) {
				cout << "   From " << nodeTypeToString((CuNodeType)i) << " to " << nodeTypeToString((CuNodeType)j) << ": " << numOfEdges << endl;
			}
		}
	}
}


void dumpEdges(NodeId node, CuEdges *edges, std::shared_ptr<CuProblem> problem)
{
	for (EdgeId e(0); e < edges->getNumOfOutEdges(node); e++) {
		NodeId targetNode = edges->getTargetNode(node, e);
		cout << "  --> " << (short)targetNode << " (Weight=" << edges->getWeight(node, e);
		cout << ")" << endl;
	}
}


void CuConstructionGraph::dumpDecisionNet(std::shared_ptr<CuProblem> problem)
{
	copyToHost();

	cout << endl << "Decision-Net:" << endl;

	cout << "Nodes:" << endl;

	for (NodeId i = NodeId(0); i < NodeId(nodes.getNumOfNodes()); i++) {
		CuNodeType nodeType = nodes.getNodeType(i);
		switch (nodeType) {
		case CuNodeType::RootNode:
			cout << "* " << (short)i << " RootNode" << endl;
			dumpEdges(i, edges, problem);
			break;
		case CuNodeType::VehTypeDepotNode:
			cout << "* " << (short)i << " VehTypeDepotNode";
			cout << " (Depot = " << nodes.getPayloadId(i);
			cout << ", Vehicle = " << (short)nodes.getVehTypeId(i) << ")" << endl;
			dumpEdges(i, edges, problem);
			break;
		case CuNodeType::ServiceTripNode: {
			cout << "* " << (short)i << " ServiceTripNode" << endl;
			ServiceTripId id = ServiceTripId(nodes.getPayloadId(i));
			CuServiceTrip trip = problem->getServiceTrips().getServiceTrip(id);
			cout << "  Id:        " << (short)id << endl;
			cout << "  From:      " << (short)trip.fromStopId << endl;
			cout << "  To:        " << (short)trip.toStopId << endl;
			cout << "  Departure: " << (int)trip.departure << endl;
			cout << "  Arrival:   " << (int)trip.arrival << endl;
			dumpEdges(i, edges, problem);
			break;
		}
		case CuNodeType::ChargingStationNode:
			cout << "* " << (short)i << " ChargingStationNode" << endl;
			cout << "  --> Payload-Id:     " << nodes.getPayloadId(i) << endl;
			cout << "  --> VehicleType-Id: " << (short)nodes.getVehTypeId(i) << endl;
			dumpEdges(i, edges, problem);
			break;
		}
	}
}


/*
Überprüft jede Kante des Entscheidungsnetzes. Sind alle Kanten plausibel? Sind alle Knoten erreichbar?
*/
void CuConstructionGraph::check(std::shared_ptr<CuProblem> problem)
{
	copyToHost();

	CuVector1<bool> isReachable(nodes.getNumOfNodes());
	//isReachable.unsetAll();

	for (NodeId n = NodeId(0); n < NodeId(nodes.getNumOfNodes()); n++) {
		CuNodeType src_type = nodes.getNodeType(n);

		int numOfEdges = edges->getNumOfOutEdges(n);

		for (EdgeId edgeId(0); edgeId < EdgeId(numOfEdges); edgeId++) {
			NodeId trg_node = edges->getTargetNode(n, edgeId);
			CuNodeType trg_type = nodes.getNodeType(trg_node);
			isReachable[(short)trg_node] = true;

			switch (src_type) {
			case CuNodeType::ChargingStationNode: {
				assert(trg_type == CuNodeType::ServiceTripNode);
				StopId stop = StopId(nodes.getPayloadId(n));
				ServiceTripId st_id = ServiceTripId(nodes.getPayloadId(trg_node));
				const CuServiceTrip& servTrip = problem->getServiceTrips().getServiceTrip(st_id);
				if (!problem->getConnectionMatrix().connected(stop, servTrip.fromStopId)) {
					cerr << "FEHLER: ChargingStationNode -> ServiceTripNode: Fahrt nicht möglich!" << endl;
				}
				assert(numOfEdges >= 1);
				break;
			}
			case CuNodeType::RootNode: {
				assert(numOfEdges >= 1);
				assert(trg_type == CuNodeType::VehTypeDepotNode);
				break;
			}
			case CuNodeType::ServiceTripNode:
			{
				assert(numOfEdges >= 1);
				if (trg_type == CuNodeType::VehTypeDepotNode) {
					ServiceTripId st_id = ServiceTripId(nodes.getPayloadId(n));
					CuServiceTrip servTrip = problem->getServiceTrips().getServiceTrip(st_id);
					StopId stop1 = servTrip.toStopId;
					StopId stop2 = StopId(nodes.getPayloadId(trg_node));
					if (!problem->getConnectionMatrix().connected(stop1, stop2)) {
						cerr << "FEHLER: ServiceTripNode --> VehTypeDepotNode: Fahrt nicht möglich!" << endl;
					}

					if (!problem->getVehicleTypeGroups().hasVehicleType(servTrip.vehicleTypeGroupId, nodes.getVehTypeId(trg_node))) {
						cerr << "FEHLER: ServiceTripNode --> VehTypeDepotNode: Unpassende Fahrzeugtypgruppe!" << endl;
					}
				}
				else {
					ServiceTripId st_id = ServiceTripId(nodes.getPayloadId(n));
					CuServiceTrip servTrip = problem->getServiceTrips().getServiceTrip(st_id);
					StopId stop1 = servTrip.toStopId;
					StopId stop2;
					if (trg_type == CuNodeType::ServiceTripNode) {
						st_id = ServiceTripId(nodes.getPayloadId(trg_node));
						servTrip = problem->getServiceTrips().getServiceTrip(st_id);
						stop2 = servTrip.fromStopId;
						if (!problem->getConnectionMatrix().connected(stop1, stop2)) {
							cerr << "FEHLER: ServiceTripNode --> ServiceTripNode: Fahrt nicht möglich!" << endl;
						}
					}
					else if (trg_type == CuNodeType::ChargingStationNode) {
						stop2 = StopId(nodes.getPayloadId(trg_node));
						if (!problem->getConnectionMatrix().connected(stop1, stop2)) {
							cerr << "FEHLER: ServiceTripNode --> ChargingStationNode: Fahrt nicht möglich!" << endl;
						}
					}
					else {
						assert(false); // Zielknoten vom falschen Typ!
					}
				}
				break;
			}
			case CuNodeType::VehTypeDepotNode: {
				assert(trg_type == CuNodeType::ServiceTripNode);
				StopId depot = StopId(nodes.getPayloadId(n));
				ServiceTripId st_id = ServiceTripId(nodes.getPayloadId(trg_node));
				const CuServiceTrip& servTrip = problem->getServiceTrips().getServiceTrip(st_id);
				if (!problem->getConnectionMatrix().connected(depot, servTrip.fromStopId)) {
					cerr << "FEHLER: VehTypeDepotNode -> ServiceTripNode: Fahrt nicht möglich!" << endl;
				}
				assert(numOfEdges >= 1);
				break;
			}
			default:
				assert(false);
			}
		}
	}

	for (int n = 1; n < nodes.getNumOfNodes(); n++)
	{
		if (!isReachable.get(n)) {
			cerr << "Warnung: Der Knoten " << n << " ist nicht erreichbar (Typ: " << nodeTypeToString(nodes.getNodeType(NodeId(n))) << ", Payload: " << nodes.getPayloadId(NodeId(n)) << ")!" << endl;
		}
	}
}

