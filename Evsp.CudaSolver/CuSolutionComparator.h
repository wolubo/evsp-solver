#pragma once

#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuSolutionComparator.h"
#include "CuEvaluationResult.h"
#include "CuComparatorResult.h"
#include "CuPlans.h"
#include "CuConstructionGraph.h"
#include "CuProblem.h"


/// <summary>
/// Vergleichen der bewerteten Lösungen.
/// </summary>
class CuSolutionComparator
{
public:
	CuSolutionComparator(int populationSize, PlattformConfig plattform);
	~CuSolutionComparator();
	
	shared_ptr<CuComparatorResult> run(shared_ptr<CuEvaluationResult> evaluationResult);
	bool check(const CuComparatorResult &comparatorResult, const CuEvaluationResult &evaluationResult);

	//shared_ptr<CuComparatorResult> getResults() const { return _comparatorResults; }

protected:
	shared_ptr<CuComparatorResult> runOnCpu(shared_ptr<CuEvaluationResult> evaluationResult);
	shared_ptr<CuComparatorResult> runOnGpu(shared_ptr<CuEvaluationResult> evaluationResult);
	int _populationSize;
	shared_ptr<CuComparatorResult> _comparatorResults;
	PlattformConfig _plattform;
};

