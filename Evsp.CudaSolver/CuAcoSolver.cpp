#include "CuAcoSolver.h"
#include "RandomGpu.h"
#include "CuSolutionGeneratorAcoCpu.h"
#include "CuSolutionGeneratorAcoGpu.h"


CuAcoSolver::CuAcoSolver(shared_ptr<CuProblem> problem, shared_ptr<CuConstructionGraph> ptn, AcoParams acoParams, AcoQualifiers qualifiers, 
	PlattformConfig plattform, int numOfThreads, bool verbose, shared_ptr<ResultLogger> resultLogger)
	: CuSolver(resultLogger), _problem(problem), _chanceOfRandomSelection(0.0f), _fading(qualifiers.fading), 
	_acoQualifiers(qualifiers),
	_selectionResult(new CuSelectionResult(acoParams.populationSize, acoParams.maxNumOfCirculations, acoParams.maxCirculationLength)), _verbose(verbose), _generator(), _normalizeEdgeWeights(acoParams.normalizeEdgeWeights),
	_evaluator(ptn, problem, plattform, acoParams.populationSize, acoParams.maxNumOfCirculations),
	_comparator(acoParams.populationSize, plattform),
	_selector(ptn, acoParams.dumpBestSolution, acoParams.dumpWorstSolution, plattform, acoParams.populationSize, acoParams.maxNumOfCirculations, acoParams.maxCirculationLength),
	_preparator(acoParams.populationSize, ptn, qualifiers, plattform, numOfThreads)
{
	switch (plattform) {
	case CPU: {
		_generator = make_shared<CuSolutionGeneratorAcoCpu>(problem, ptn, acoParams.populationSize, acoParams.maxNumOfCirculations, acoParams.maxCirculationLength, numOfThreads, acoParams.chargeLevel, verbose, acoParams.keepBestSolution);
		break;
	}
	case GPU: {
		shared_ptr<RandomGpu> randGpu = make_shared<RandomGpu>(acoParams.populationSize);
		_generator = make_shared<CuSolutionGeneratorAcoGpu>(problem, ptn, acoParams.populationSize, acoParams.maxNumOfCirculations, acoParams.maxCirculationLength, acoParams.chargeLevel, randGpu, verbose, acoParams.keepBestSolution);
		break;
	}
	default:
		throw std::logic_error("Unbekannter Wert für PlattformConfig!");
	}
}


CuAcoSolver::~CuAcoSolver()
{
}


shared_ptr<CuSolution> CuAcoSolver::getSolution()
{
	//assert(false);
	return 0;
}


void CuAcoSolver::loop()
{
	shared_ptr<CuPlans> solutions = _generator->run(_selectionResult, _chanceOfRandomSelection);// Generieren einer neuen (bzw. ersten) Generation.
	shared_ptr<CuEvaluationResult> evaluationResult = _evaluator.run(solutions);			// Bewerten der generierten Lösungen.
	shared_ptr<CuComparatorResult> comparatorResult = _comparator.run(evaluationResult);	// Vergleicht die bewerteten Lösungen.
	_selectionResult = _selector.run(evaluationResult, comparatorResult, solutions);		// Auswahl der besten Lösung/-en.

	if (_selectionResult->hasNewBestSolution()) {
		_chanceOfRandomSelection = 0.0f;		
		_fading = _acoQualifiers.fading;

		addResult("NEUE LÖSUNG", _selectionResult->getBestSolution().getTotalCost(), _selectionResult->getBestSolution().getNumOfCirculations());
	}
	else {
		_chanceOfRandomSelection += 0.001f;
		if (_chanceOfRandomSelection > 0.2f) _chanceOfRandomSelection = 0.2f;

		//_fading *= 1.05f;
		//if (_fading > 1.0f) _fading = 1.0f;
	}

	_preparator.run(solutions, evaluationResult, _selectionResult, comparatorResult, _fading, _normalizeEdgeWeights); // Vorbereiten der nächsten Generation.
}


void CuAcoSolver::setup()
{
	_chanceOfRandomSelection = 0.0f;
}


void CuAcoSolver::teardown()
{
	//int solId = _selectionResult->getBestSolution().getSolutionId();
	//int numOfVehicles = bestSolution->numOfVehicles;
	//int totalCost = (int)bestSolution->totalCost;
	//std::cout << "ERGEBNIS: " << "Gesamtkosten=" << totalCost << ", Fahrzeuganzahl=" << numOfVehicles << endl;
}

