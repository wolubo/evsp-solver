#include "Solutions.h"
#include <iomanip>
#include <thread>
#include <mutex>
#include <algorithm>
#include "EVSP.BaseClasses/Stopwatch.h"
#include "RandomCpu.h"


struct Solutions::StatsData {
	int highestNumOfCirculations = 0;
	int highestNumOfActions = 0;
	int lowestNumOfCirculations = INT_MAX;
	int lowestNumOfActions = INT_MAX;
};


Solutions::Solutions(int numOfThreads, shared_ptr<CuProblem> problem, SaParams params)
	: _solutions()
{
	assert(problem);
	assert(numOfThreads >= 1);

	//numOfThreads = 1;

	Stopwatch stopwatch;
	stopwatch.start();

	int numOfServiceTrips = problem->getServiceTrips().getNumOfServiceTrips();

	_solutions.reserve(params.populationSize);
	for (int i = 0; i < params.populationSize; i++) {
		_solutions.push_back(Solution(params.maxCirculationLength, numOfServiceTrips));
	}

	if (params.populationSize < numOfThreads * 10) numOfThreads = params.populationSize / 10; // TODO Grenzwert konfigurierbar machen.

	shared_ptr<int> errorCounter = make_shared<int>(0);
	shared_ptr<StatsData> stats = make_shared<StatsData>();

	if (numOfThreads > 1) {
		int solutionsPerThead = (params.populationSize + numOfThreads - 1) / numOfThreads;

		thread *t = new thread[numOfThreads];

		int fromSolId, toSolId;

		for (int i = 0; i < numOfThreads; ++i) {
			fromSolId = i * solutionsPerThead;
			toSolId = (i + 1) * solutionsPerThead - 1;
			if (toSolId >= params.populationSize) toSolId = params.populationSize - 1;
			t[i] = thread(createSolutions, std::ref(_solutions), fromSolId, toSolId, problem, errorCounter, params, numOfServiceTrips, stats);
		}

		for (int i = 0; i < numOfThreads; ++i) {
			t[i].join();
		}
	}
	else {
		createSolutions(std::ref(_solutions), 0, params.populationSize - 1, problem, errorCounter, params, numOfServiceTrips, stats);
	}

	if (params.printStats) {
		cout << "Startlösung generiert:" << endl;
		cout << "Anzahl der Umläufe in einer Lösung:  min=" << stats->lowestNumOfCirculations << std::endl;
		cout << "                                     max=" << stats->highestNumOfCirculations << std::endl;
		cout << "Anzahl der Aktionen in einem Umlauf: min=" << stats->lowestNumOfActions << std::endl;
		cout << "                                     max=" << stats->highestNumOfActions << std::endl;
	}

	if (*errorCounter > 0) {
		cerr << "Achtung: " << *errorCounter << " Lösungen sind fehlerhaft!" << endl;
	}

	stopwatch.stop("Start-Lösungen generiert (Greedy, CPU): ");
}


Solutions::Solutions(const Solutions &other)
	: _solutions(other._solutions)
{
}


Solutions::~Solutions()
{
}


std::mutex createSolutions_mutex;


void Solutions::createSolutions(vector<Solution> &solutions, int fromSolutionId, int toSolutionId, shared_ptr<CuProblem> problem, shared_ptr<int> errorCounter, SaParams params, int numOfServiceTrips, shared_ptr<StatsData> stats)
{
	RandomCpu randCpu;
	for (int solutionId = fromSolutionId; solutionId <= toSolutionId; solutionId++) {
		Solution &solution = solutions.at(solutionId);
		shared_ptr<CuVector1<int>> rndServiceTrips = randCpu.shuffle(numOfServiceTrips);
		for (int i = 0; i < numOfServiceTrips; i++) {
			ServiceTripId stId = ServiceTripId(rndServiceTrips->get(i));
			if (params.greedyCreation) {
				solution.insertServiceTripGreedy(stId, problem);
			}
			else {
				solution.insertServiceTripRandom(stId, problem, randCpu);
			}
		}

		if (params.printStats) {
			StatsData localStats;

			SolutionStats solStats = solution.getStats(problem);
			localStats.lowestNumOfCirculations = std::min(solStats.getNumOfCirculations(), localStats.lowestNumOfCirculations);
			localStats.highestNumOfCirculations = std::max(solStats.getNumOfCirculations(), localStats.highestNumOfCirculations);
			localStats.lowestNumOfActions = std::min(solStats.getLowestNumOfActions(), localStats.lowestNumOfActions);
			localStats.highestNumOfActions = std::max(solStats.getHighestNumOfActions(), localStats.highestNumOfActions);

			createSolutions_mutex.lock();
			stats->lowestNumOfCirculations = std::min(localStats.lowestNumOfCirculations, stats->lowestNumOfCirculations);
			stats->highestNumOfCirculations = std::max(localStats.highestNumOfCirculations, stats->highestNumOfCirculations);
			stats->lowestNumOfActions = std::min(localStats.lowestNumOfActions, stats->lowestNumOfActions);
			stats->highestNumOfActions = std::max(localStats.highestNumOfActions, stats->highestNumOfActions);
			createSolutions_mutex.unlock();
		}

		if (params.performChecks && !solution.check(problem)) {
			createSolutions_mutex.lock();
			cerr << "Fehler in Lösung " << solutionId << endl;
			(*errorCounter)++;
			createSolutions_mutex.unlock();
		}
	}
}


bool Solutions::operator==(Solutions &rhs)
{
	if (this == &rhs) return true;

	if (getNumOfSolutions() != rhs.getNumOfSolutions()) return false;

	for (int solId = 0; solId < getNumOfSolutions(); solId++) {
		if (getSolution(solId) != rhs.getSolution(solId)) return false;
	}

	return true;
}


bool Solutions::operator!=(Solutions &rhs)
{
	return !(*this == rhs);
}


Solution& Solutions::getSolution(int solutionId)
{
	assert(solutionId >= 0);
	assert(solutionId < getNumOfSolutions());
	return _solutions.at(solutionId);
}


SolutionKeyData Solutions::getBestSolution(shared_ptr<CuProblem> problem)
{
	SolutionKeyData retVal(-1, AmountOfMoney(INT_MAX), INT_MAX);

	vector<Solution>::iterator iter = _solutions.begin();
	int solutionId = 0;

	while (iter != _solutions.end()) { // TODO Schleife parallelisieren.
		Solution &solution = *iter;
		SolutionStats stats = solution.getStats(problem);
		if (stats.getTotalCost() < retVal.getTotalCost()) {
			retVal = SolutionKeyData(solutionId, stats.getTotalCost(), stats.getNumOfCirculations());
		}
		solutionId++;
		iter++;
	}

	//cout << "Best in class: " << retVal.getSolutionId() << " mit " << (int)retVal.getTotalCost() << endl;

	return retVal;
}


void Solutions::dump()
{
	for (int i = 0; i < getNumOfSolutions(); i++) {
		cout << "Solution " << i << ":" << endl;
		_solutions.at(i).dump();
		cout << endl;
	}
}


void Solutions::mutateSolutions(SaParams params, int numOfThreads, shared_ptr<CuProblem> problem, Temperature currentTemperature)
{
	Stopwatch stopwatch;
	stopwatch.start();

	int populationSize = (int)_solutions.size();

	if (populationSize < numOfThreads * 10) numOfThreads = populationSize / 10; // TODO Grenzwert konfigurierbar machen.

	//numOfThreads = 1;

	shared_ptr<int> errorCounter = make_shared<int>(0);
	shared_ptr<int> betterClone = make_shared<int>(0);
	shared_ptr<int> cloneRules = make_shared<int>(0);

	if (numOfThreads > 1) {
		int solutionsPerThead = (populationSize + numOfThreads - 1) / numOfThreads;

		thread *t = new thread[numOfThreads];

		int fromSolId, toSolId;

		for (int i = 0; i < numOfThreads; ++i) {
			fromSolId = i * solutionsPerThead;
			toSolId = (i + 1) * solutionsPerThead - 1;
			if (toSolId >= populationSize) toSolId = populationSize - 1;
			t[i] = thread(createMutations, std::ref(_solutions), fromSolId, toSolId, problem, errorCounter, params, currentTemperature, betterClone, cloneRules);
		}

		for (int i = 0; i < numOfThreads; ++i) {
			t[i].join();
		}
	}
	else {
		createMutations(std::ref(_solutions), 0, populationSize - 1, problem, errorCounter, params, currentTemperature, std::ref(betterClone), std::ref(cloneRules));
	}

	if (*errorCounter > 0) {
		cerr << "Achtung: " << *errorCounter << " Lösungen sind fehlerhaft!" << endl;
	}

	if (params.printStats) {
		cout << "Mutationen erzeugt: Erfolgsrate=" << setprecision(1) << ((float)*betterClone / (float)populationSize)*100.0f << "%, " << "Überschreibungsrate=" << ((float)*cloneRules / (float)populationSize)*100.0f << "%" << endl;
	}

	stopwatch.stop("Mutierte Lösungen generiert (CPU): ");
}


std::mutex createMutations_mutex;


void Solutions::createMutations(vector<Solution> &solutions, int fromSolId, int toSolId, shared_ptr<CuProblem> problem, shared_ptr<int> errorCounter, SaParams params, Temperature currentTemperature, shared_ptr<int> betterClone, shared_ptr<int> cloneRules)
{
	int l_betterClone = 0, l_cloneRules = 0;

	RandomCpu randCpu;
	int numOfServiceTrips = problem->getServiceTrips().getNumOfServiceTrips();
	for (int solutionId = fromSolId; solutionId <= toSolId; solutionId++) {

		Solution &original = solutions.at(solutionId);
		Solution clone(original);

		// Zufällige Auswahl der durchzuführenden Mutation.
		const int numOfOperations = 5;

		CuVector1<float> chances(numOfOperations);
		chances[0] = params.crossoverChance;
		chances[1] = params.insertionChance;
		chances[2] = params.circCostDeletionChance;
		chances[3] = params.numOfServiceTripsDeletionChance;
		chances[4] = params.randomDeletionChance;

		int operation = randCpu.weightedRandomSelection(5, chances);

		switch (operation) {
		case 0: {
			assert(numOfOperations > 4);
			clone.performCrossoverOperation(numOfServiceTrips, problem, params.crossoverRate, params.crossoverUpperBound, randCpu);
			break;
		}
		case 1: {
			assert(numOfOperations > 4);
			bool greedy = randCpu.shot(params.greedyInsertion); // Greedy insertion?
			clone.performInsertionOperation(greedy, numOfServiceTrips, params.insertionRate, params.insertionUpperBound, problem, randCpu);
			break;
		}
		case 2: {
			assert(numOfOperations > 4);
			bool greedy = randCpu.shot(params.greedyInsertionAfterDeletion); // Greedy insertion?
			clone.performDeleteOperation(params.deletionRate, params.deletionsLowerBound, params.deletionsUpperBound,
				Solution::DeletionMode::CIRC_COST_RATIO, greedy, problem, randCpu);
			break;
		}
		case 3: {
			assert(numOfOperations > 4);
			bool greedy = randCpu.shot(params.greedyInsertionAfterDeletion); // Greedy insertion?
			clone.performDeleteOperation(params.deletionRate, params.deletionsLowerBound, params.deletionsUpperBound,
				Solution::DeletionMode::NUM_OF_SERVICE_TRIPS, greedy, problem, randCpu);
			break;
		}
		case 4: {
			assert(numOfOperations > 4);
			bool greedy = randCpu.shot(params.greedyInsertionAfterDeletion); // Greedy insertion?
			clone.performDeleteOperation(params.deletionRate, params.deletionsLowerBound, params.deletionsUpperBound,
				Solution::DeletionMode::RANDOM, greedy, problem, randCpu);
			break;
		}
		default:
			assert(false);
		}

		// Prüfe den Clone, falls 'check' gesetzt ist.
		if (params.performChecks && !clone.check(problem)) {
			createMutations_mutex.lock();
			cerr << "Fehler im Clone " << solutionId << endl;
			(*errorCounter)++;
			createMutations_mutex.unlock();
			break;
		}

		// Bewerte das Original und den Clone. 
		SolutionStats originalStats = original.getStats(problem);
		SolutionStats cloneStats = clone.getStats(problem);
		bool cloneWins = cloneStats.getTotalCost() < originalStats.getTotalCost();

		if (cloneWins) {
			l_betterClone++;
		}
		else {
			// Der Clone ist schlechter als das Original. Entscheide, ob der Clone trotzdem übernommen werden soll.

			// Wahrscheinlichkeit dafür bestimmen, dass der Clone überlebt.
			float deltaE = (float)cloneStats.getTotalCost() - (float)originalStats.getTotalCost();
			assert(deltaE >= 0.0f);

			float t = (float)currentTemperature;
			assert(t > 0.0f);
			float p = exp(-(deltaE / t));	// TODO Faktor einführen, der entweder deltaE oder t beeinflusst (Multiplikator, ggf. auch Offset).
			cloneWins = randCpu.shot(p);

			if (cloneWins) l_cloneRules++;
		}

		if (cloneWins) {
			Solution &target = solutions.at(solutionId);
			target = clone;
		}

	}

	// Statistische Daten sichern
	createMutations_mutex.lock();
	*betterClone += l_betterClone;
	*cloneRules += l_cloneRules;
	createMutations_mutex.unlock();
}
