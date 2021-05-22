#pragma once

#include "EVSP.CudaSolver/CuSolver.h"
#include "EVSP.BaseClasses/Configuration.h"
#include "EVSP.Model/Problem.h"


class SolverFabric
{
public:
	SolverFabric(shared_ptr<Configuration> config, shared_ptr<Problem> problem);
	~SolverFabric();

	std::shared_ptr<CuSolver> createSaSolver();
	std::shared_ptr<CuSolver> createAcoSolver();

private:
	std::shared_ptr<CuProblem> convertProblem(shared_ptr<Problem> problem, PlattformConfig plattform);
	shared_ptr<CuProblem> _problem;
	shared_ptr<Configuration> _config;
};

