#pragma once

#ifdef __CUDACC__
#include "cuda_runtime.h"
#include "RandomGpu.h"
#else
#include "RandomCpu.h"
#endif

#include <limits.h>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuMatrix1.hpp"
#include "CuLockVector1.h"
#include "CuConstructionGraph.h"
#include "CuEdges.h"
#include "CuNodes.h"
#include "CuProblem.h"
#include "CuPlans.h"
#include "MatrixCreator.h"
#include "CuVector2.hpp"
#include "CuBoolVector1.hpp"
#include "AntsRaceState.h"
#include "PrepareEdgeSelection.h"


#ifdef __CUDACC__
CU_DEV EdgeId selectEdge(int solutionId, NodeId currNodeId, CuVector1<EdgeId> *activeEdges, CuVector1<float> *weights, CuConstructionGraph *ptn,
	AntsRaceState *state, CuVector1<bool> *activityStateOfNodes, RandomGpu *rand, CuLockVector1 *nodeLock,
	bool verbose, float chanceOfRandomSelection)
#else
EdgeId selectEdge(int solutionId, NodeId currNodeId, CuVector1<EdgeId> *activeEdges, CuVector1<float> *weights, CuConstructionGraph *ptn,
	AntsRaceState *state, CuVector1<bool> *activityStateOfNodes, RandomCpu *rand, CuLockVector1 *nodeLock,
	bool verbose, float chanceOfRandomSelection)
#endif
{
	float totalWeight = 0.0f;

#ifdef __CUDACC__
	int numOfActiveEdges = prepareEdgeSelectionGpu(activeEdges, ptn, currNodeId, activityStateOfNodes, state, weights, totalWeight, nodeLock);
#else
	int numOfActiveEdges = prepareEdgeSelection(activeEdges, ptn, currNodeId, activityStateOfNodes, state, weights, totalWeight);
#endif	

	if (numOfActiveEdges == 0) {
		if (verbose) printf("Selection: Nothing!\n");
		return EdgeId::invalid();
	}

	// Falls nur eine einzige aktive Kante existiert, so wird diese ausgewählt.
	// Existieren mehrere aktive Kanten, so erfolgt eine zufällige Auswahl.
	ushort edgeIndex = 0;
	if (numOfActiveEdges > 1) {
		// Mit einer gewissen Wahrscheinlichkeit erfolgt die Kantenauswahl unabhängig vom Kantengewicht.
		// Ausserdem passiert dies, wenn die Summe der Kantengewichte der aktiven Kanten unterhalb 
		// einer kritischen Schwelle liegt.
		// TODO Wahrscheinlichkeitswert und Schwelle konfigurierbar machen.

		float limit = numOfActiveEdges * FLT_EPSILON * 10.0f;
		assert(limit > 0.0f);

		assert(chanceOfRandomSelection >= 0.0f);
		assert(chanceOfRandomSelection <= 1.0f);

#ifdef __CUDACC__
		if (totalWeight < limit || rand->shot(chanceOfRandomSelection, solutionId))
		{
			edgeIndex = rand->rand(numOfActiveEdges, solutionId);
		}
		else {
			// Andernfalls erfolgt eine gewichete Zufallsauswahl.
			edgeIndex = rand->weightedRandomSelection(numOfActiveEdges, *weights, totalWeight, solutionId);
		}
#else
		if (totalWeight < limit || rand->shot(chanceOfRandomSelection))
		{
			edgeIndex = rand->rand(numOfActiveEdges);
		}
		else {
			// Andernfalls erfolgt eine gewichete Zufallsauswahl.
			edgeIndex = rand->weightedRandomSelection(numOfActiveEdges, *weights, totalWeight);
		}
#endif
	}

	assert(edgeIndex < numOfActiveEdges);
	EdgeId selectedEdgeId = (*activeEdges)[edgeIndex];
	if (verbose) printf("Selection: %i (Node=%i)\n", (short)selectedEdgeId, (short)ptn->edges->getTargetNode(currNodeId, selectedEdgeId));
	return selectedEdgeId;
}


#ifdef __CUDACC__
CU_DEV bool createCirculation(int solutionId, CuPlans *solutions, CuConstructionGraph *ptn, AntsRaceState *state, CuVector1<EdgeId> *activeEdges,
	CuVector1<float> *weights, CuVector1<bool> *activityStateOfNodes, int &numOfRemainingServTrips,
	RandomGpu *rand, bool verbose, CuLockVector1 *nodeLock, float chanceOfRandomSelection)
#else
bool createCirculation(int solutionId, CuPlans *solutions, CuConstructionGraph *ptn, AntsRaceState *state, CuVector1<EdgeId> *activeEdges,
	CuVector1<float> *weights, CuVector1<bool> *activityStateOfNodes, int &numOfRemainingServTrips,
	RandomCpu *rand, bool verbose, CuLockVector1 *nodeLock, float chanceOfRandomSelection)
#endif
{
	NodeId currNodeId = NodeId(0); // Id des aktuellen Knotens. Es beginnt mit dem Wurzelknoten.
	EdgeId selectedEdgeId;

	assert(ptn->nodes.getNodeType(currNodeId) == CuNodeType::RootNode);

	CirculationId currentCirculationId;

	do {
		assert(currNodeId == 0);
		if (verbose) printf("currNodeId=%i\n", (short)currNodeId);
		if (verbose) printf("*** Visit RootNode! Remaining Servicetrips: %i\n", numOfRemainingServTrips);

		// Wähle das Depot und den Fahrzeugtypen.
		selectedEdgeId = selectEdge(solutionId, currNodeId, activeEdges, weights, ptn, state, activityStateOfNodes, rand, nodeLock, verbose, chanceOfRandomSelection);
		if (!selectedEdgeId.isValid()) {
			//state->initialize();
			return false; // Es kann kein neuer Umlauf gebildet werden!
		}
		currNodeId = ptn->edges->getTargetNode(currNodeId, selectedEdgeId);
		if (verbose) printf("currNodeId=%i\n", (short)currNodeId);
		assert(ptn->nodes.getNodeType(currNodeId) == CuNodeType::VehTypeDepotNode);

		StopId depotId = StopId(ptn->nodes.getPayloadId(currNodeId));
		VehicleTypeId vehTypeId = ptn->nodes.getVehTypeId(currNodeId);
		state->startCirculation(depotId, vehTypeId);

		// Wähle die erste Servicefahrt.
		selectedEdgeId = selectEdge(solutionId, currNodeId, activeEdges, weights, ptn, state, activityStateOfNodes, rand, nodeLock, verbose, chanceOfRandomSelection);
		if (selectedEdgeId.isValid()) {
			if (verbose) printf("*** Visit VehTypeDepotNode %i: depot=%i, vehType=%i\n", (short)currNodeId, (short)state->getStartDepotId(), (short)state->getVehTypeId());
			currentCirculationId = solutions->addNewCirculation(solutionId, depotId, vehTypeId);
			solutions->appendNode(solutionId, currentCirculationId, currNodeId, selectedEdgeId);
			currNodeId = ptn->edges->getTargetNode(currNodeId, selectedEdgeId);
			assert(ptn->nodes.getNodeType(currNodeId) == CuNodeType::ServiceTripNode);
		}
		else {
			if (verbose) printf("*** Ignore VehTypeDepotNode %i: depot=%i, vehType=%i\n", (short)currNodeId, (short)state->getStartDepotId(), (short)state->getVehTypeId());
			activityStateOfNodes->set((short)currNodeId, false); // Knoten deaktivieren, da von diesem Knoten keine SF abgehen.
			currNodeId = NodeId(0);
		}

	} while (!currentCirculationId.isValid());

	bool keepOnGoing = true;
	do {
		if (verbose) printf("currNodeId=%i\n", (short)currNodeId);

		CuNodeType nodeType = ptn->nodes.getNodeType(currNodeId);
		if (nodeType == CuNodeType::VehTypeDepotNode) {
			if (verbose) printf("*** Visit VehTypeDepotNode %i: Depot: %i,  Aktuelle Haltestelle=%i\n", (short)currNodeId, (short)state->getStartDepotId(), (short)state->getCurrentStopId());
			state->processEmptyTrip(state->getStartDepotId());
		}
		else if (nodeType == CuNodeType::ChargingStationNode) {
			StopId stopId = StopId(ptn->nodes.getPayloadId(currNodeId)); // Haltestellen-Id der mit der Node assoziierten Ladestation.
			if (verbose) printf("*** Visit ChargingStationNode %i (stopId=%i)!\n", (short)currNodeId, (short)stopId);
			state->processCharging(stopId);
		}
		else if (nodeType == CuNodeType::ServiceTripNode) {
			ServiceTripId servTripId = ServiceTripId(ptn->nodes.getPayloadId(currNodeId));
			if (verbose) printf("*** Visit ServiceTripNode %i! ServTripId=%i\n", (short)currNodeId, (short)servTripId);
			state->processServiceTrip(servTripId);
			activityStateOfNodes->set((short)currNodeId, false);
			assert(numOfRemainingServTrips > 0);
			numOfRemainingServTrips--;
		}
		else {
			assert(false); // Unbekannter oder hier nicht erwarteter NodeType.
		}

		selectedEdgeId = selectEdge(solutionId, currNodeId, activeEdges, weights, ptn, state, activityStateOfNodes, rand, nodeLock, verbose, chanceOfRandomSelection);

		if (!selectedEdgeId.isValid()) {
			if (nodeType != CuNodeType::VehTypeDepotNode) {
				printf("FEHLER: Fahrzeug an Haltestelle gestrandet (fehlende Einrückfahrt)!\n");
			}
			keepOnGoing = false;
		}

		solutions->appendNode(solutionId, currentCirculationId, currNodeId, selectedEdgeId);

		if (keepOnGoing) {
			currNodeId = ptn->edges->getTargetNode(currNodeId, selectedEdgeId);
		}

	} while (keepOnGoing);

	state->endCirculation();

	return true;
}

#ifdef __CUDACC__
CU_DEV void createPlanGpu(int solutionId, CuPlans *solutions, CuConstructionGraph *ptn, AntsRaceState *state, CuVector1<EdgeId> *activeEdges, CuVector1<float> *weights, CuVector1<bool> *activityStateOfNodes, int &numOfRemainingServTrips,
	RandomGpu *rand, bool verbose, CuLockVector1 *nodeLock, float chanceOfRandomSelection)
#else
void createPlanCpu(int solutionId, CuPlans *solutions, CuConstructionGraph *ptn, AntsRaceState *state, CuVector1<EdgeId> *activeEdges, CuVector1<float> *weights, CuVector1<bool> *activityStateOfNodes, int &numOfRemainingServTrips,
	RandomCpu *rand, bool verbose, float chanceOfRandomSelection)
#endif
{
	solutions->reInit(solutionId);

	bool keepOnGoing = true;
	do {
#ifdef __CUDACC__
		keepOnGoing = createCirculation(solutionId, solutions, ptn, state, activeEdges, weights, activityStateOfNodes, numOfRemainingServTrips, rand, verbose, nodeLock, chanceOfRandomSelection);
#else
		keepOnGoing = createCirculation(solutionId, solutions, ptn, state, activeEdges, weights, activityStateOfNodes, numOfRemainingServTrips, rand, verbose, 0, chanceOfRandomSelection);
#endif
	} while (numOfRemainingServTrips > 0 && keepOnGoing);
}

