#pragma once

#include "device_launch_parameters.h"
#include "Temperature.hpp"
#include "CuLockVector1.h"
#include "CuConstructionGraph.h"
#include "CuEdges.h"
#include "CuNodes.h"
#include "CuProblem.h"
#include "CuPlans.h"
#include "MatrixCreator.h"
#include "RandomGpu.h"
#include "CuVector2.hpp"
#include "CuVector1.hpp"
#include "AntsRaceState.h"
#include "CuSelectionResult.h"
#include "EVSP.BaseClasses/Typedefs.h"

/// <summary>
/// </summary>
/// <param name="populationSize"></param>
/// <param name="ptn_dev"></param>
/// <param name="problem_dev"></param>
/// <param name="solutions_dev"></param>
/// <param name="rand_dev"></param>
/// <param name="isNodeActiveMat_dev"></param>
/// <param name="consumption"></param>
/// <param name="chargeLevel">Prozentwert zwischen 0.0 (0%) und 1.0 (100%), der angibt, ab welchen Batterieladestand das Aufsuchen von Ladestationen erlaubt ist.</param>
/// <param name="allActiveEdges">Ids der aktiven Kanten. Ein Vektor für jeden Kernel. Wird nur innerhalb des Kernels benutzt.</param>
/// <param name="allWeights">Gewichtung der aktiven Kanten. Ein Vektor für jeden Kernel. Wird nur innerhalb des Kernels benutzt.</param>
__global__ void antsRaceKernel(int populationSize, CuSelectionResult *selectionResult, CuConstructionGraph *ptn_dev, CuProblem *problem_dev,
	CuPlans *solutions_dev, RandomGpu *rand_dev, CuVector1<CuVector1<bool>*> *isNodeActiveMat_dev,
	ConsumptionMatrix *consumption, float chargeLevel, CuVector1<CuVector1<EdgeId>*> *allActiveEdges, 
	CuVector1<CuVector1<float>*> *allWeights, CuLockVector1 *nodeLock, bool verbose, float chanceOfRandomSelection,
	bool keepBestSolution);

