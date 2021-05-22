#pragma once

#include <time.h>
#include <memory>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/Stopwatch.h"
#include "CuSolutionGenerator.h"
#include "CuSolutionEvaluatorAco.h"
#include "CuSolutionComparator.h"
#include "CuSolutionSelectorAco.h"
#include "CuSolutionPreparatorAco.h"
#include "ResultLogger.h"

using namespace std;

class CuProblem;
class CuSolution;

typedef bool(TerminationDelegate)();

/// <summary>
///
/// </summary>
class CuSolver {
public:
	static bool setupCuda();
	static void resetCuda();

	CuSolver(shared_ptr<ResultLogger> resultLogger);
	CuSolver(CuSolver&) = delete;
	~CuSolver();

	/// <summary>
	/// Führt alle zur Lösung des Problems nötigen Schritte durch.
	/// </summary>
	/// <returns>ProblemSolution-Objekt, das alle Daten zur gefundenen Lösung enthält.</returns>
	void run();

	/// <summary>
	/// Liefert die Lösung zurück.
	/// </summary>
	/// <returns>ProblemSolution-Objekt, das alle Daten zur gefundenen Lösung enthält.</returns>
	virtual shared_ptr<CuSolution> getSolution() = 0;

	/// <summary>
	/// Definiert eine externe Funktion, die nach jeder Runde aufgerufen werden soll. Liefert diese Funktion 'true',
	/// so terminiert die Suche.
	/// </summary>
	/// <param name="terminationDelegate"></param>
	/// <returns></returns>
	void setTerminationDelegate(TerminationDelegate terminationDelegate);

	/// <summary>
	/// Legt eine Rundenanzahl fest, nach der die Suche terminieren soll. Default: 1000
	/// Bei '0' wird die Rundenanzahl als Terminierungskriterium nicht berücksichtigt.
	/// </summary>
	/// <param name="maxNumOfRounds"></param>
	/// <returns></returns>
	void setMaxNumOfRounds(int maxNumOfRounds);

	int getRoundCounter() const { return _roundCounter; }

	/// <summary>
	/// Legt einen Zeitraum in Sekunden fest, nach dem die Suche terminieren soll. Default: 1000
	/// Bei Werten <= 0.0 wird der Zeitraum als Terminierungskriterium nicht berücksichtigt.
	/// </summary>
	/// <param name="maxElapsedTime"></param>
	/// <returns></returns>
	void setMaxElapsedTime(float maxElapsedTime);

	/// <summary>
	/// Liefert die seit dem Start von run() vergangene Zeit.
	/// </summary>
	float getElapsedSeconds();

	/// <summary>
	/// Liefert die seit dem Start von run() vergangene Zeit als String.
	/// <param name="numOfDigits">Anzahl der Nachkommastellen.</param>
	/// </summary>
	std::string getElapsedSecondsAsStr(int numOfDigits);

protected:
	virtual void loop() = 0;
	virtual void setup() = 0;
	virtual void teardown() = 0;

	void addResult(string caption, AmountOfMoney totalCost, int numOfVehicles);

private:
	shared_ptr<ResultLogger> _resultLogger;

	/// <summary>
	/// Prüft, ob mindestens ein Abbruchkriterium erfüllt ist.
	/// </summary>
	/// <returns>TRUE, wenn die Abbruchkriterien erfüllt sind.</returns>
	bool checkTerminationConditions();

	void increaseRoundCounter();

	/// <summary>
	/// run() soll terminieren, sobald dieses Delegate TRUE liefert.
	/// </summary>
	TerminationDelegate *_terminationDelegate;

	/// <summary>
	/// run() soll nach der hier definierten Anzahl von Runden terminieren (Default: 1000).
	/// </summary>
	int _maxNumOfRounds;

	/// <summary>
	/// run() soll nach der hier definierten Anzahl von Sekunden terminieren (Default: 1000).
	/// </summary>
	float _maxElapsedTime;

	Stopwatch _stopwatch;

	int _roundCounter;
};
