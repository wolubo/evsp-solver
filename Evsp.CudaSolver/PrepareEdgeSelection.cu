#include "PrepareEdgeSelection.h"

#include "cuda_runtime.h"

/*
Sammle alle aktiven Kanten des aktuellen Knotens.
Eine Kante ist aktiv, wenn sie auf einen aktiven Knoten verweist.
Kanten zu Servicefahrten werden nur dann als aktiv angesehen, wenn die Servicefahrt mit dem aktuellen Fahrzeug
durchführbar ist.
Kanten zu Ladestationen werden nur dann als aktiv angesehen, wenn eine gewisse Batteriekapazität unterschritten ist.
Falls die Mindestbatteriekapazität unterschritten ist gelten nur noch Kanten zum Startdepot und zu Ladestationen als aktiv.

Wenn keine aktiven Kanten gefunden werden können richtet sich die weitere Verarbeitung danach, von welchem Typ der aktuelle
Knoten ist:
- 'VehTypeDepotNode' oder 'ChargingStationNode': Der aktuelle Knoten wird deaktiviert, ohne besucht zu werden und die Verarbeitung kehrt zum vorhergehenden Knoten zurück. Der Entscheidungspfad wird entsprechend korrigiert.
- 'RootNode': Die Verarbeitung wird beendet. KLÄREN: KANN DAS AUFTRETEN?
- 'ServiceTripNode': Eine Fehlermeldung wird ausgegeben, denn diese Situation darf nicht auftreten. Zumindest die Kante
zur EndNode muss gefunden werden.
- 'EndNode': Eine Fehlermeldung wird ausgegeben, denn diese Situation darf nicht auftreten. Zumindest die Kante
zur RootNode muss gefunden werden.
*/


CU_HSTDEV bool checkChargingTrip(CuConstructionGraph *ptn, NodeId chargingNodeId, StopId chargingStationId, CuVector1<bool> *activityStateOfNodes, AntsRaceState *state);


CU_HSTDEV bool selection(EdgeId edgeIdx, CuConstructionGraph *ptn, NodeId currNodeId, CuVector1<bool> *activityStateOfNodes, AntsRaceState *state)
{
	NodeId targetNodeId;
	CuNodeType sourceNodeType, targetNodeType;
	bool addEdgeToSelection = false;
	sourceNodeType = ptn->nodes.getNodeType(currNodeId);
	targetNodeId = ptn->edges->getTargetNode(currNodeId, edgeIdx);
	targetNodeType = ptn->nodes.getNodeType(targetNodeId);

	switch (targetNodeType)
	{
	case  CuNodeType::VehTypeDepotNode:
	{
		// TODO Klären: Vorrang für Ladestation vor Einrückfahrt, falls Mindestbatteriekapazität unterschritten ist?
		// TODO Prüfen, ob die Batteriekapazität ausreicht, um das Depot zu erreichen? 
		// TODO Oder vorab max. Entfernung zu Depots ermitteln und Mindestkapazität so festlegen, dass Depot immer erreicht wird?
		// TODO Evtl. Beschränkung von Depots auf bestimmte Fahrzeugtypen einbauen. Kapazitätsgrenzen von Depots einbauen.
		StopId depotId = StopId(ptn->nodes.getPayloadId(targetNodeId));
		VehicleTypeId vehTypeId = ptn->nodes.getVehTypeId(targetNodeId);
		StopId startDepot = state->getStartDepotId();
		VehicleTypeId currVehTypeId = state->getVehTypeId();
		if (startDepot.isValid()) {
			if (sourceNodeType == CuNodeType::RootNode) {
				addEdgeToSelection = activityStateOfNodes->get((short)targetNodeId);
			}
			else if (sourceNodeType == CuNodeType::ServiceTripNode) {
				// Falls der Umlauf bereits begonnen hat: Nur die Rückfahrt ins Startdepot ist erlaubt.
				// Damit der "richtige" Knoten gewählt wird muss zusätzlich der Fahrzeugtyp verglichen werden.
				addEdgeToSelection = startDepot == depotId && vehTypeId == currVehTypeId;

				// Aussertdem muss eine ggf. nötige Verbindungsfahrt auch möglich sein.
				if (addEdgeToSelection) {
					addEdgeToSelection = state->canProcessEmptyTrip(startDepot);
					if (!addEdgeToSelection) {
						printf("Verbindungsfahrt ins Depot %i ist nicht möglich!\n", (short)startDepot);
					}
				}
			}
			else {
				assert(false);
			}
		}
		else {
			// Falls ein neuer Umlauf beginnt: Nur die Fahrt in aktive Depots (von denen noch Servicefahrten abgehen)
			// ist erlaubt.
			addEdgeToSelection = activityStateOfNodes->get((short)targetNodeId);
		}
		break;
	}
	case CuNodeType::ChargingStationNode:
	{
		if (activityStateOfNodes->get((short)targetNodeId)) {
			// Prüfen, ob das Aufladen sinnvoll ist und ob es nach der Ladestation überhaupt weiter gehen kann (ob es 
			// also von dort aus noch absolvierbare Servicefahrten gibt).
			StopId chargingStationId = StopId(ptn->nodes.getPayloadId(targetNodeId));
			if (state->canProcessCharging(chargingStationId)) {
				addEdgeToSelection = checkChargingTrip(ptn, targetNodeId, chargingStationId, activityStateOfNodes, state);
			}
		}
		break;
	}
	case CuNodeType::ServiceTripNode:
	{
		if (activityStateOfNodes->get((short)targetNodeId)) {
			addEdgeToSelection = state->canProcessServiceTrip(ServiceTripId(ptn->nodes.getPayloadId(targetNodeId)));
		}
		break;
	}
	case CuNodeType::RootNode:
		assert(false);
		break;
	default:
		assert(false);
	}

	return addEdgeToSelection;
}


/* Simuliert die Fahrt zu einer Ladestation und die Aufladung dort. Liefert TRUE, wenn es nach dem Aufladen von der
Ladestation aus absolvierbare Servicefahrten gibt. */
CU_HSTDEV bool checkChargingTrip(CuConstructionGraph *ptn, NodeId chargingNodeId, StopId chargingStationId, CuVector1<bool> *activityStateOfNodes, AntsRaceState *state)
{
	AntsRaceState cloneState(*state);
	cloneState.processCharging(chargingStationId);
	for (EdgeId edge(0); edge < ptn->edges->getNumOfOutEdges(chargingNodeId); edge++) {
		if (selection(edge, ptn, chargingNodeId, activityStateOfNodes, &cloneState)) {
			// Es existiert wenigstens eine Aktion, die nach dem Aufladen absolviert werden kann. Das Fahrzeug wird also 
			// nicht an der Ladestation stranden, weil es von dort aus keine Weiterfahrt gibt..
			return true;
		}
	}
	return false;
}


CU_HSTDEV int prepareEdgeSelection(CuVector1<EdgeId> *activeEdges, CuConstructionGraph *ptn, NodeId currNodeId, CuVector1<bool> *activityStateOfNodes,
	AntsRaceState *state, CuVector1<float> *weights, float &totalWeight)
{
	int numOfActiveEdges = 0;

	for (EdgeId edgeIdx(0); edgeIdx < EdgeId(ptn->edges->getNumOfOutEdges(currNodeId)); edgeIdx++) {
		if (selection(edgeIdx, ptn, currNodeId, activityStateOfNodes, state)) {
			float edgeWeight = ptn->edges->getWeight(currNodeId, edgeIdx);
			(*activeEdges)[numOfActiveEdges] = edgeIdx;
			(*weights)[numOfActiveEdges] = edgeWeight;
			totalWeight += (*weights)[numOfActiveEdges];
			numOfActiveEdges++;
			assert(numOfActiveEdges < Max_EdgesPerNode);
		}
	}
	return numOfActiveEdges;
}


__global__ void prepareEdgeSelectionKernel(CuVector1<EdgeId> *activeEdges, CuConstructionGraph *ptn, NodeId currNodeId, CuVector1<bool> *activityStateOfNodes, AntsRaceState *state, CuVector1<float> *weights, float *totalWeight, CuLockVector1 *nodeLock, int *numOfActiveEdges)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tx >= ptn->edges->getNumOfOutEdges(currNodeId)) return;
	EdgeId edgeIdx = EdgeId(tx);

	bool addEdgeToSelection = false;

	addEdgeToSelection = selection(edgeIdx, ptn, currNodeId, activityStateOfNodes, state);

	float edgeWeight = ptn->edges->getWeight(currNodeId, edgeIdx);

	if (addEdgeToSelection) {
		int newId = -1;
		bool success = false;
		do {
			if (nodeLock->lock((short)currNodeId)) {
				newId = *numOfActiveEdges;
				*totalWeight += edgeWeight;
				(*numOfActiveEdges)++;
				assert(*numOfActiveEdges < Max_EdgesPerNode);
				success = true;
				nodeLock->unlock((short)currNodeId);
			}
		} while (!success);
		(*activeEdges)[newId] = edgeIdx;
		(*weights)[newId] = edgeWeight;
	}
}


CU_DEV int prepareEdgeSelectionGpu(CuVector1<EdgeId> *activeEdges, CuConstructionGraph *ptn, NodeId currNodeId, CuVector1<bool> *activityStateOfNodes, AntsRaceState *state, CuVector1<float> *weights, float &totalWeight, CuLockVector1 *nodeLock)
{
	int numOfActiveEdges = 0;
	int *numOfActiveEdgesPtr;
	float *totalWeightPtr;
	int numOfOutEdges = ptn->edges->getNumOfOutEdges(currNodeId);

	numOfActiveEdgesPtr = (int*)malloc(sizeof(int));
	*numOfActiveEdgesPtr = 0;
	totalWeightPtr = (float*)malloc(sizeof(float));
	*totalWeightPtr = totalWeight;

	int blockSize = 1024;
	int numOfBlocks = (numOfOutEdges + blockSize - 1) / blockSize;
	dim3 dimGrid(numOfBlocks);
	dim3 dimBlock(blockSize);

	prepareEdgeSelectionKernel << <dimGrid, dimBlock >> > (activeEdges, ptn, currNodeId, activityStateOfNodes, state, weights, totalWeightPtr, nodeLock, numOfActiveEdgesPtr);
	CUDA_CHECK_DEV(cudaGetLastError());
	CUDA_CHECK_DEV(cudaDeviceSynchronize());

	numOfActiveEdges = *numOfActiveEdgesPtr;
	totalWeight = *totalWeightPtr;

	free(numOfActiveEdgesPtr);
	free(totalWeightPtr);

	return numOfActiveEdges;
}
