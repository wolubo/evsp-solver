#include "CuSolutionPreparatorAcoKernels.h"


__global__ void updatePheromoneTracksKernel(CuMatrix1<float> *newWeight, CuMatrix1<int> *counter, CuEdges *edges, float fading, 
	float avgWeight, bool newAllTimesBest, int numOfNodes)
{
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x; // Node-Id
	int tid_y = blockIdx.y * blockDim.y + threadIdx.y; // Edge-Index

	if (tid_x >= numOfNodes) return;
	if (tid_y >= edges->getNumOfOutEdges(NodeId(tid_x))) return;

	//int oldVisitCounter = edges->getVisitCounter(NodeId(tid_x), EdgeId(tid_y));

	float oldWeight;
	if (newAllTimesBest) {
		// In der letzten Runde wurde ein neuer Bestwert gefunden. Deshalb die alten Gewichte "vergessen", damit die neue Lösung greift.
		oldWeight = 0.0f;
	}
	else {
		oldWeight = edges->getWeight(NodeId(tid_x), EdgeId(tid_y));
		oldWeight *= fading; // Verblassen der bisherigen Spur berücksichtigen.
	}

	float newTrack = newWeight->get(tid_x, tid_y);
	int newVisits = counter->get(tid_x, tid_y);

	float newValue;
	if (newVisits > 0) {
		newValue = oldWeight + newTrack;
	}
	else {
		newValue = oldWeight + avgWeight;
	}

	edges->setWeight(NodeId(tid_x), EdgeId(tid_y), newValue);
}


__global__ void collectPheromonesKernel(int solutionId, int numOfCirculations, int numOfSteps, CuMatrix1<float> *newWeight, CuMatrix1<int> *counter, CuPlans *solutions, float newValue, CuLockMatrix1 *lock)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx >= numOfCirculations) return;
	if (ty >= numOfSteps) return;

	CirculationId circId(tx);
	assert(circId.isValid());

	if (ty >= solutions->getNumOfNodes(solutionId, circId)) return;

	CircStepIndex stepId(ty);
	assert(stepId.isValid());

	EdgeId edgeId = solutions->getEdgeId(solutionId, circId, stepId);
	if (!edgeId.isValid()) return;

	NodeId currentNodeId = solutions->getNodeId(solutionId, circId, stepId);
	assert(currentNodeId.isValid());

	// Zellen in counter und newWeight gegen den Zugriff durch andere Threads sperren und die hinterlassene Pheromon-Menge addieren.
	bool success = false;
	do {
		if (lock->lock((short)currentNodeId, (short)edgeId)) {

			// Spur auf die bereits gesammelten Spuren addieren.
			float & fValue = newWeight->itemAt((short)currentNodeId, (short)edgeId);
			fValue += newValue;

			// Counter inkrementieren.
			int &cValue = counter->itemAt((short)currentNodeId, (short)edgeId);
			cValue++;

			lock->unlock((short)currentNodeId, (short)edgeId);
			success = true;
		}
	} while (!success);
}

