#pragma once

#include "RandomCpu.h"
#include "CuServiceTrips.h"
#include "CuProblem.h"
#include "Circulation.h"
#include "Solution.h"
#include "MatrixCreator.h"
#include "SolutionKeyData.h"
#include "Temperature.hpp"

using namespace std;

/// <summary>
/// Enth�lt s�mtliche Uml�ufe der gesamten Population. 
/// </summary>
class Solutions
{
public:
	/// <summary>
	/// Erzeugt eine Population von L�sungen.
	/// </summary>
	/// <param name="greedy">TRUE: Erzeuge m�glichst kosteng�nstige L�sungen. FALSE: Erzeuge L�sungen nach dem Zufallsprinzip.</param>
	Solutions(int numOfThreads, shared_ptr<CuProblem> problem, SaParams params);

	Solutions(const Solutions &other);

	~Solutions();

	Solutions& operator=(const Solutions &rhs) { assert(false); }

	bool operator==( Solutions &rhs);
	bool operator!=( Solutions &rhs);

	void mutateSolutions(SaParams params, int numOfThreads, shared_ptr<CuProblem> problem, Temperature currentTemperature);

	/// <summary>
	/// Liefert die Anzahl der L�sungen.
	/// </summary>
	int getNumOfSolutions() const { return (int)_solutions.size(); }

	Solution& getSolution(int solutionId);

	SolutionKeyData getBestSolution(shared_ptr<CuProblem> problem);

	void dump();

private:
	struct StatsData;
	static void createSolutions(vector<Solution> &solutions, int fromSolutionId, int toSolutionId, shared_ptr<CuProblem> problem, shared_ptr<int> errorCounter, SaParams params, int numOfServiceTrips, shared_ptr<StatsData> stats);
	static void createMutations(vector<Solution> &solutions, int fromSolId, int toSolId, shared_ptr<CuProblem> problem, shared_ptr<int> errorCounter, SaParams params, Temperature currentTemperature, shared_ptr<int> betterClone, shared_ptr<int> cloneRules);
	vector<Solution> _solutions;
};
