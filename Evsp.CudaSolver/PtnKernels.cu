#include "PtnKernels.h"

#include "CudaCheck.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include "EVSP.BaseClasses/Typedefs.h"
#include "PtnCreateEdges.hpp"


__global__ void createPtnEdgesKernel(CuConstructionGraph *ptn_dev, CuLockVector1 *lock, CuProblem *problem_dev, VehicleTypeGroupIntersection* vtgIntersect_dev, float initialWeight, int numOfNodes)
{
	int threadId_x = blockIdx.x * blockDim.x + threadIdx.x;
	int threadId_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (threadId_x >= numOfNodes || threadId_y >= numOfNodes) return; // Block kann grösser sein.
	if (threadId_x == threadId_y) return; // Kein Knoten referenziert sich selbst.

	NodeId sourceNodeId(threadId_x);
	NodeId targetNodeId(threadId_y);

	handleNodePair(sourceNodeId, targetNodeId, ptn_dev, lock, problem_dev, vtgIntersect_dev, initialWeight, numOfNodes);
}
