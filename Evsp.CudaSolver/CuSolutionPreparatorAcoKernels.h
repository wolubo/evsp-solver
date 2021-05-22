#pragma once

#include "cuda_runtime.h"

#include "EVSP.BaseClasses/Typedefs.h"
#include "CuVector1.hpp"
#include "CuMatrix1.hpp"
#include "CuEdges.h"
#include "CuPlans.h"
#include "CuLockMatrix1.h"


__global__ void collectPheromonesKernel(int solutionId, int numOfCirculations, int numOfSteps, CuMatrix1<float> *newWeight, CuMatrix1<int> *counter, CuPlans *solutions, float newValue, CuLockMatrix1 *lock);

__global__ void updatePheromoneTracksKernel(CuMatrix1<float> *newWeight, CuMatrix1<int> *counter, CuEdges *edges,
	float fading, float avgWeight, bool newAllTimesBest, int numOfNodes);
