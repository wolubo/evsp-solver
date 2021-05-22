#include "PtnCreateEdgesCpu.h"
#include "PtnCreateEdges.hpp"

void createPtnEdgesOnCpu(CuConstructionGraph *ptn, std::shared_ptr<CuProblem> problem, shared_ptr<VehicleTypeGroupIntersection> vtgIntersect, float initialWeight, int numberOfNodes)
{
	for (NodeId sourceNodeId(0); sourceNodeId < numberOfNodes; sourceNodeId++) {
		for (NodeId targetNodeId(0); targetNodeId < numberOfNodes; targetNodeId++) {
			handleNodePair(sourceNodeId, targetNodeId, ptn, problem.get(), vtgIntersect.get(), initialWeight, numberOfNodes);
		}
	}
}
