#pragma once

#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuSolutionEvaluatorAco.h"
#include "CuSolutionGeneratorAcoGpu.h"
#include "CuEvaluationResult.h"


/// <summary>
/// Bewerten der generierten Lösungen.
/// </summary>
class CuSolutionEvaluatorAco
{
public:
	CuSolutionEvaluatorAco(shared_ptr<CuConstructionGraph> ptn, shared_ptr<CuProblem> problem, PlattformConfig plattform, int populationSize, int maxNumOfCirculations);
	~CuSolutionEvaluatorAco();
	
	shared_ptr<CuEvaluationResult> run(shared_ptr <CuPlans> solutions);

	//shared_ptr<CuPlans> getSolutions() const { return _solutions; }
	//shared_ptr<CuEvaluationResult> getResults() const { return _results; }

	static CU_HSTDEV void analyseSolution(int solutionId, CuPlans *plans, CuConstructionGraph *ptn, CuProblem *problem, CuEvaluationResult *result);

protected:
	struct State {
		CU_HSTDEV State();
		StopId currStop;
		AmountOfMoney totalNetCost;
		AmountOfMoney totalGrossCost;
		PointInTime currentTime;
		PointInTime circulationStartTime;
	};

	shared_ptr<CuEvaluationResult> runOnCpu(shared_ptr <CuPlans> solutions);
	shared_ptr<CuEvaluationResult> runOnGpu(shared_ptr <CuPlans> solutions);

	static CU_HSTDEV DurationInSeconds calculateEmptyTrip(StopId from, StopId to, State &state, CuProblem *problem, CuVehicleType &vehicleType);
	static CU_HSTDEV void analyseCirculation(int solutionId, CirculationId circulationId, CuPlans *plans, CuConstructionGraph *ptn, 
		CuProblem *problem, CuEvaluationResult *result, AmountOfMoney &netCostSum, AmountOfMoney &grossCostSum);
	static CU_HSTDEV void analyseCirculationStep(NodeId nodeId, EdgeId edgeId, State &state, CuConstructionGraph *ptn, CuProblem *problem, CuVehicleType &vehicleType);
	shared_ptr<CuEvaluationResult> _results;
	shared_ptr<CuConstructionGraph> _ptn;
	shared_ptr<CuProblem> _problem;
	PlattformConfig _plattform;
};

