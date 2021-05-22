#pragma once

#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuPlans.h"
#include "CuProblem.h"
#include "CuConstructionGraph.h"
#include "CuSelectionResult.h"
#include "CuSolutionGenerator.h"

/// <summary>
/// Erzeugt eine neue (bzw. erste) Generation von Lösungen.
/// </summary>
class CuSolutionGeneratorAcoGpu : public CuSolutionGenerator
{
public:
	/// <summary>
	/// </summary>
	/// <param name="chargeLevel">Prozentwert zwischen 0.0 (0%) und 1.0 (100%), der angibt, ab welchen Batterieladestand das Aufsuchen von Ladestationen erlaubt ist.</param>
	CuSolutionGeneratorAcoGpu(std::shared_ptr<CuProblem> problem, std::shared_ptr<CuConstructionGraph> ptn, int populationSize,
		int maxNumOfCirculations, int maxNumOfNodes, float chargeLevel, shared_ptr<RandomGpu> rand, bool verbose, bool keepBestSolution);

	~CuSolutionGeneratorAcoGpu();

	virtual shared_ptr<CuPlans> run(shared_ptr<CuSelectionResult> selectionResult, float chanceOfRandomSelection);

protected:
	shared_ptr<RandomGpu> _rand;

	// Prozentwert zwischen 0.0 (0%) und 1.0 (100%), der angibt, ab welchen Batterieladestand das Aufsuchen von Ladestationen erlaubt ist.
	float _chargeLevel;
};

