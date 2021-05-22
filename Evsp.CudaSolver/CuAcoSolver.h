#pragma once

#include "CuSolver.h"

class CuAcoSolver :	public CuSolver
{
public:
	CuAcoSolver(shared_ptr<CuProblem> problem, shared_ptr<CuConstructionGraph> ptn, AcoParams acoParams, AcoQualifiers qualifiers, 
		PlattformConfig plattform, int numOfThreads, bool verbose, shared_ptr<ResultLogger> resultLogger);
	~CuAcoSolver();

	virtual shared_ptr<CuSolution> getSolution();

private:
	virtual void loop();
	virtual void setup();
	virtual void teardown();

	bool _verbose;

	std::shared_ptr<CuProblem> _problem;

	shared_ptr<CuSelectionResult> _selectionResult;

	shared_ptr<CuSolutionGenerator>	_generator;
	CuSolutionEvaluatorAco	_evaluator;
	CuSolutionComparator _comparator;
	CuSolutionSelectorAco	_selector;
	CuSolutionPreparatorAco _preparator;

	bool _normalizeEdgeWeights;
	float _chanceOfRandomSelection;
	float _fading;
	AcoQualifiers _acoQualifiers;
};

