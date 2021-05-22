#pragma once

#include <mutex>
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuProblem.h"
#include "CuConstructionGraph.h"
#include "CuLockMatrix1.h"
#include "CuPlans.h"
#include "CuEvaluationResult.h"
#include "CuSelectionResult.h"
#include "CuComparatorResult.h"

/// <summary>
/// Vorbereiten der nächsten Generation.
/// </summary>
class CuSolutionPreparatorAco
{
public:
	CuSolutionPreparatorAco(int populationSize, shared_ptr<CuConstructionGraph> ptn, AcoQualifiers qualifiers, PlattformConfig plattform, int numOfThreads);
	~CuSolutionPreparatorAco();

	void run(shared_ptr<CuPlans> solutions, shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuSelectionResult> selectResult, shared_ptr<CuComparatorResult> compResult, float fading, bool normalizeEdgeWeights);

protected:
	void runOnCpu(shared_ptr<CuPlans> solutions, shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuSelectionResult> selectResult, shared_ptr<CuComparatorResult> compResult, float fading, bool normalizeEdgeWeights);
	void runOnGpu(shared_ptr<CuPlans> solutions, shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuSelectionResult> selectResult, shared_ptr<CuComparatorResult> compResult, float fading, bool normalizeEdgeWeights);

	static void updatePheromoneTracksOnCpu(int fromSolId, int toSolId, CuSolutionPreparatorAco *p, std::mutex &theMutex, shared_ptr<CuPlans> solutions, shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuSelectionResult> selectResult, shared_ptr<CuComparatorResult> compResult);

	// Normiert alle Kantengewichte auf das Intervall [0, initialWeight].
	void CuSolutionPreparatorAco::performNormalizeEdgeWeights(shared_ptr<CuPlans> solutions);

	int _populationSize;
	AcoQualifiers _acoQualifiers;
	shared_ptr<CuConstructionGraph> _ptn;
	CuLockMatrix1 *_lock;
	PlattformConfig _plattform;
	int _numOfThreads;
};

