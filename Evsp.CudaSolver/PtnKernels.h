#pragma once

#include "CuLockVector1.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuConstructionGraph.h"
#include "CuEdges.h"
#include "CuNodes.h"
#include "CuProblem.h"
#include "MatrixCreator.h"


__global__ void createPtnEdgesKernel(CuConstructionGraph *ptn_dev, CuLockVector1 *lock, CuProblem *problem_dev, VehicleTypeGroupIntersection* vtgIntersect_dev, float initialWeight, int numOfNodes);
