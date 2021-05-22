#pragma once

#include <memory>
#include "CuPlans.h"
#include "CuProblem.h"
#include "CuConstructionGraph.h"
#include "CuSelectionResult.h"

class CuSolutionGenerator
{
public:
	CuSolutionGenerator(std::shared_ptr<CuProblem> problem, std::shared_ptr<CuConstructionGraph> ptn, int populationSize, int maxNumOfCirculations, int maxNumOfNodes, bool verbose, bool keepBestSolution);

	virtual shared_ptr<CuPlans> run(shared_ptr<CuSelectionResult> selectionResult, float chanceOfRandomSelection) = 0;

protected:

	/// <summary>
	/// 
	/// </summary>
	int _populationSize;

	/// <summary>
	/// 
	/// </summary>
	shared_ptr<CuPlans> _plans;

	/// <summary>
	/// 
	/// </summary>
	int _maxNumOfNodes;

	/// <summary>
	/// 
	/// </summary>
	std::shared_ptr<CuProblem> _problem;

	/// <summary>
	/// 
	/// </summary>
	std::shared_ptr<CuConstructionGraph> _ptn;

	std::shared_ptr<ConsumptionMatrix> _batteryConsumption;

	bool _verbose;

	bool _keepBestSolution;
};

