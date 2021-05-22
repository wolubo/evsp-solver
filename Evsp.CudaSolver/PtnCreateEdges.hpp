#pragma once

#ifdef __CUDACC__
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "CudaCheck.h"
#include "device_launch_parameters.h"
#else
#include <mutex>
#endif

#include "CuLockVector1.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuConstructionGraph.h"
#include "CuEdges.h"
#include "CuNodes.h"
#include "CuProblem.h"
#include "MatrixCreator.h"


#ifdef __CUDACC__

__device__ void addEdge(CuLockVector1 *lock, CuEdges &edges, NodeId fromNode, NodeId toNode, float initWeight)
{
	bool success = false;
	do {
		if (lock->lock((short)fromNode)) {
			edges.addEdge(fromNode, toNode, initWeight);
			lock->unlock((short)fromNode);
			success = true;
		}
	} while (!success);
}

#else

std::mutex addEdgeMutex;

void addEdge(void *lock, CuEdges &edges, NodeId fromNode, NodeId toNode, float initWeight)
{
	addEdgeMutex.lock();
	edges.addEdge(fromNode, toNode, initWeight);
	addEdgeMutex.unlock();
}

#endif


#ifdef __CUDACC__
__device__ void handleNodePair(NodeId sourceNodeId, NodeId targetNodeId, CuConstructionGraph *ptn, CuLockVector1 *lock, CuProblem *problem, VehicleTypeGroupIntersection* vtgIntersect, float initialWeight, int numOfNodes)
#else
void handleNodePair(NodeId sourceNodeId, NodeId targetNodeId, CuConstructionGraph *ptn, CuProblem *problem, VehicleTypeGroupIntersection* vtgIntersect, float initialWeight, int numOfNodes)
#endif
{
#ifndef __CUDACC__
	void *lock = 0;
#endif;

	ushort sourcePayloadId = ptn->nodes.getPayloadId(sourceNodeId);
	ushort targetPayloadId = ptn->nodes.getPayloadId(targetNodeId);

	CuNodeType targetNodeType = ptn->nodes.getNodeType(targetNodeId);

	switch (ptn->nodes.getNodeType(sourceNodeId)) {
	case CuNodeType::RootNode:
	{
		if (targetNodeType == CuNodeType::VehTypeDepotNode)
		{	// Auf den Wurzelknoten folgen nur Depots. Alle anderen Knoten ignorieren.
			// Der Wurzelknoten ist mit allen Depotknoten verbunden.
			addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, initialWeight);
		}
		break;
	}
	case CuNodeType::VehTypeDepotNode:
	{
		if (targetNodeType == CuNodeType::ServiceTripNode)
		{
			// Auf einen Fahrzeugtyp-Depotknoten folgen nur Servicefahrtknoten. Alle anderen Knoten ignorieren.

			// TODO Depots enthalten nur bestimmte Fahrzeugtypen? Dann hier filtern!

			StopId depot_id = StopId(sourcePayloadId);
			VehicleTypeId vehType_id = ptn->nodes.getVehTypeId(sourceNodeId);
			ServiceTripId servTrip_id = ServiceTripId(targetPayloadId);
			const CuServiceTrip &servTrip = problem->getServiceTrips().getServiceTrip(servTrip_id);

			// Initialgewicht der Servicefahrt berechnen.
			int earliest = (int)problem->getServiceTrips().getEarliestStartTime();
			int latest = (int)problem->getServiceTrips().getLatestStartTime();
			float servTripWeight;
			if (earliest != latest) {
				servTripWeight = (initialWeight * 0.1f) + ((float)(latest - (int)servTrip.arrival) / (latest - earliest)) * initialWeight * 0.9f;
			}
			else {
				servTripWeight = initialWeight;
			}

			// Kann die Servicefahrt von diesem Fahrzeugtypen bedient werden?
			if (problem->getVehicleTypeGroups().hasVehicleType(servTrip.vehicleTypeGroupId, vehType_id))
			{
				// Gibt es eine Verbindungsfahrt vom Depot zur Starthaltestelle?
				if (depot_id == servTrip.fromStopId) {
					addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, servTripWeight);
				}
				else {
					if (problem->getConnectionMatrix().connected(depot_id, servTrip.fromStopId)) {
						addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, servTripWeight);
					}
				}
			}
		}
		break;
	}
	case CuNodeType::ChargingStationNode:
	{
		StopId stopId = StopId(sourcePayloadId);
		if (targetNodeType == CuNodeType::ServiceTripNode)
		{
			// Nach dem Aufladen muss eine Servicefahrt folgen.
			ServiceTripId serviceTripId = ServiceTripId(targetPayloadId);
			const CuServiceTrip &servTrip = problem->getServiceTrips().getServiceTrip(serviceTripId);
			if (stopId == servTrip.fromStopId)
			{
				addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, initialWeight);
			}
			else {
				// Die beiden Haltestellen sind nicht identisch. Gibt es eine Verbindungsfahrt?
				if (problem->getConnectionMatrix().connected(stopId, servTrip.fromStopId)) {
					// Es existiert eine Verbindungsfahrt.
					addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, initialWeight);
				}
			}
		}
		break;
	}
	case CuNodeType::ServiceTripNode:
	{
		ServiceTripId servTripA_id = ServiceTripId(sourcePayloadId);
		const CuServiceTrip &servTripA = problem->getServiceTrips().getServiceTrip(servTripA_id);

		switch (ptn->nodes.getNodeType(targetNodeId)) {

		case CuNodeType::RootNode:
			// Wird ignoriert: Es kann keine Kante zurück zur Wurzel geben.
			break;

		case CuNodeType::ServiceTripNode: {
			ServiceTripId servTripB_id = ServiceTripId(targetPayloadId);
			const CuServiceTrip &servTripB = problem->getServiceTrips().getServiceTrip(servTripB_id);

			// Gibt es mindestens einen Fahrzeugtypen, der beide Fahrten bedienen kann?
			VehicleTypeGroupId tripA_vtg = servTripA.vehicleTypeGroupId;
			VehicleTypeGroupId tripB_vtg = servTripB.vehicleTypeGroupId;
			if (vtgIntersect->get((short)tripA_vtg, (short)tripB_vtg) != true) {
				printf("Kein gemeinsamer Fahrzeugtyp\n");
				return;
			}

			if (servTripA.arrival >= servTripB.departure) return; // Fahrt A endet erst, nachdem Fahrt B begonnen hat: Keine Kante von A nach B.

			if (servTripA.toStopId == servTripB.fromStopId)
			{
				// Trip B startet direkt an der Endhaltestelle von Trip A. Keine Verbindungsfahrt nötig.
				addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, initialWeight);
			}
			else
			{
				// Trip B startet nicht an der Endhaltestelle von Trip A. Es ist eine Verbindungsfahrt nötig.
				// TODO PrevNextMatrix nutzen?
				EmptyTripId emptyTripId = problem->getConnectionMatrix().getEmptyTripId(servTripA.toStopId, servTripB.fromStopId);
				if (!emptyTripId.isValid()) return; // Es existiert keine Verbindungsfahrt.
				const CuEmptyTrip &emptyTrip = problem->getEmptyTrips().getEmptyTrip(emptyTripId);
				DurationInSeconds spareTime = DurationInSeconds((int)servTripB.departure - (int)servTripA.arrival); // Verfügbare Zeit zwischen den beiden Fahrten.
				if (emptyTrip.duration >= spareTime) return; // Die Verbindungsfahrt dauert zu lange.
				addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, initialWeight);
			}
			break;
		}
		case CuNodeType::ChargingStationNode:
		{
			StopId stopId = StopId(targetPayloadId);
			if (servTripA.toStopId == stopId) {
				addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, initialWeight);
			}
			else {
				// Die beiden Haltestellen sind nicht identisch. Gibt es eine Verbindungsfahrt?
				if (!problem->getConnectionMatrix().connected(servTripA.toStopId, stopId)) return; // Es existiert keine Verbindungsfahrt.
				addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, initialWeight);
			}
			break;
		}
		case CuNodeType::VehTypeDepotNode:
		{
			// Nach jeder Servicefahrt kann ins Depot zurückgekehrt werden. Dort startet entweder eine noch nicht 
			// verplante Servicefahrt, oder der Umlauf endet.
			StopId stopId = StopId(targetPayloadId);

			if (servTripA.toStopId == stopId || problem->getConnectionMatrix().connected(servTripA.toStopId, stopId)) {
				VehicleTypeGroupId st_vtg = servTripA.vehicleTypeGroupId;
				VehicleTypeId depot_vt = ptn->nodes.getVehTypeId(targetNodeId);
				if (problem->getVehicleTypeGroups().hasVehicleType(st_vtg, depot_vt)) {
					addEdge(lock, *ptn->edges, sourceNodeId, targetNodeId, initialWeight);
				}
			}

			break;
		}
		default:
			assert(false); // Das darf nie passieren: Ein CuNodeType wurde hier nicht berücksichtigt.
		}
		break;
	}
	default:
		assert(false); // Das darf nie passieren: Ein CuNodeType wurde hier nicht berücksichtigt.
	}

}
