#pragma once

#include <memory>
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuEvaluationResult.h"
#include "CuComparatorResult.h"
#include "CuSolutionEvaluatorAco.h"
#include "CuPlans.h"
#include "CuPlan.h"
#include "CuSelectionResult.h"
#include "RandomCpu.h"

using namespace std;


/// <summary>
/// Auswahl der besten und ggf. der schlechtesten Lösung(en) 
/// </summary>
class CuSolutionSelectorAco
{
public:
	CuSolutionSelectorAco(shared_ptr<CuConstructionGraph> ptn, bool dumpBestSolution, bool dumpWorstSolution, PlattformConfig plattform, int populationSize, int maxNumOfCirculations, int maxNumOfNodes);
	~CuSolutionSelectorAco();
	
	shared_ptr<CuSelectionResult> run(shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuComparatorResult> compResult, shared_ptr<CuPlans> solutions);

protected:
	shared_ptr<CuSelectionResult> runOnCpu(shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuComparatorResult> compResult, shared_ptr<CuPlans> solutions);
	shared_ptr<CuSelectionResult> runOnGpu(shared_ptr<CuEvaluationResult> evalResult, shared_ptr<CuComparatorResult> compResult, shared_ptr<CuPlans> solutions);

	shared_ptr<CuConstructionGraph> _ptn;
	shared_ptr<CuSelectionResult> _selectionResult;
	bool _dumpBestSolution;
	bool _dumpWorstSolution;

	shared_ptr<RandomCpu> _rand;

	PlattformConfig _plattform;
};

