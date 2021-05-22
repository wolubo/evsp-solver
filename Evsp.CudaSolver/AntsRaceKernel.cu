#include "AntsRaceKernel.h"

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "cuda_runtime.h"
#include "CuMatrix1.hpp"
#include "CreatePlan.hpp"



__global__ void antsRaceKernel(int populationSize, CuSelectionResult *selectionResult, CuConstructionGraph *ptn, CuProblem *problem, CuPlans *solutions,	RandomGpu *rand, CuVector1<CuVector1<bool>*> *isNodeActiveMat, ConsumptionMatrix *consumption, float chargeLevel,
	CuVector1<CuVector1<EdgeId>*> *allActiveEdges, CuVector1<CuVector1<float>*> *allWeights, CuLockVector1 *nodeLock, bool verbose, float chanceOfRandomSelection, bool keepBestSolution)
{
	int solutionId = blockIdx.x * blockDim.x + threadIdx.x;
	if (solutionId >= populationSize) return;

	if (keepBestSolution && selectionResult->isAlive(solutionId)) return; // Nur inaktive ("tote") Lösungen werden ersetzt.

	CuVector1<EdgeId> *activeEdges = (*allActiveEdges)[solutionId];
	CuVector1<float> *weights = (*allWeights)[solutionId];

	CuVector1<bool> *activityStateOfNodes = (*isNodeActiveMat)[solutionId];

	// TODO CuConstructionGraph auflösen? Edges und Nodes dann direkt übergeben.
	int numOfRemainingServTrips = problem->getServiceTrips().getNumOfServiceTrips();

	AntsRaceState *state = new AntsRaceState(problem, consumption, chargeLevel, verbose); // TODO Auf Host erzeugen.

	createPlanGpu(solutionId, solutions, ptn, state, activeEdges, weights, activityStateOfNodes, numOfRemainingServTrips, rand, false,
		nodeLock, chanceOfRandomSelection);

	delete state;

	assert(numOfRemainingServTrips == 0); // Alle Servicefahrten müssen nun verplant sein.

	if (numOfRemainingServTrips > 0) {
		printf("Warnung: Es konnten nicht alle Servicefahrten eingeplant werden! Offenbar gibt es Servicefahrten, die nicht immer erreicht werden können.\n");
	}
}


