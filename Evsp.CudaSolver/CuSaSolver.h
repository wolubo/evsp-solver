#pragma once

#include "EVSP.BaseClasses/Stopwatch.h"
#include "CuSolver.h"
#include "Solutions.h"
#include "BestSolutionManager.h"

class CuSaSolver : public CuSolver
{
public:
	CuSaSolver(shared_ptr<CuProblem> problem, PlattformConfig plattform, int numOfThreads, SaParams params,
		shared_ptr<ResultLogger> resultLogger);
	~CuSaSolver();

	virtual shared_ptr<CuSolution> getSolution();

protected:
	virtual void loop();
	virtual void setup();
	virtual void teardown();
	void findBestSolution(bool printStartSolution);

private:
	shared_ptr<CuProblem> _problem;
	shared_ptr<Solutions> _solutions;
	int _numOfThreads;
	Temperature _temperature;
	SaParams _params;
	BestSolutionManager _bestSolutionManager;
};

