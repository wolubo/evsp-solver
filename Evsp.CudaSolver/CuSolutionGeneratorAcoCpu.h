#pragma once

#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuPlans.h"
#include "CuProblem.h"
#include "CuConstructionGraph.h"
#include "CuSelectionResult.h"
#include "CuSolutionGenerator.h"

/// <summary>
/// Generieren einer neuen Generation(bzw.einer ersten Generation).
/// </summary>
class CuSolutionGeneratorAcoCpu : public CuSolutionGenerator
{
public:
	/// <summary>
	/// </summary>
	/// <param name="chargeLevel">Prozentwert zwischen 0.0 (0%) und 1.0 (100%), der angibt, ab welchen Batterieladestand das Aufsuchen von Ladestationen erlaubt ist.</param>
	CuSolutionGeneratorAcoCpu(std::shared_ptr<CuProblem> problem, std::shared_ptr<CuConstructionGraph> ptn, int populationSize, int maxNumOfCirculations, int maxNumOfNodes, int numOfThreads, float chargeLevel, bool verbose, bool keepBestSolution);

	~CuSolutionGeneratorAcoCpu();

	virtual shared_ptr<CuPlans> run(shared_ptr<CuSelectionResult> selectionResult, float chanceOfRandomSelection);

	//shared_ptr<CuPlans> getSolutions();
	//shared_ptr<CuPlans> getDecisionPaths() { return _plans; }

protected:
	static void antsRace(int fromSolId, int toSolId, shared_ptr<CuSelectionResult> selectionResult, shared_ptr<CuConstructionGraph> ptn, shared_ptr<CuProblem> problem, shared_ptr<CuPlans> solutions, shared_ptr<ConsumptionMatrix> consumption, float chargeLevel, bool verbose, float chanceOfRandomSelection, bool keepBestSolution);
	int _numOfThreads;

	// Prozentwert zwischen 0.0 (0%) und 1.0 (100%), der angibt, ab welchen Batterieladestand das Aufsuchen von Ladestationen erlaubt ist.
	float _chargeLevel;
};

