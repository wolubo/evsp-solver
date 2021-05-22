#include "CuSolutionGenerator.h"


CuSolutionGenerator::CuSolutionGenerator(std::shared_ptr<CuProblem> problem, std::shared_ptr<CuConstructionGraph> ptn, int populationSize, int maxNumOfCirculations, int maxNumOfNodes, bool verbose, bool keepBestSolution)
	: _populationSize(populationSize), _problem(problem), _ptn(ptn), _maxNumOfNodes(maxNumOfNodes), _verbose(verbose), _keepBestSolution(keepBestSolution),
	_plans(new CuPlans(populationSize, maxNumOfCirculations, maxNumOfNodes)),
	_batteryConsumption( new ConsumptionMatrix(problem->getEmptyTrips(), problem->getServiceTrips(), problem->getVehicleTypes(), PlattformConfig::CPU))
{

}