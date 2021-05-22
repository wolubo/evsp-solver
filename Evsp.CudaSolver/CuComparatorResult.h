#pragma once

#include <assert.h>
#include "EVSP.BaseClasses/Typedefs.h"
#include "SolutionKeyData.h"

class CuSolutionComparator;
class CuEvaluationResult;


struct CirculationKeyData {
	CU_HSTDEV CirculationKeyData() {}
	CU_HSTDEV CirculationKeyData(int theSolutionId, CirculationId theCirculationId, float theValue) { solutionId = theSolutionId; circulationId = theCirculationId; value = theValue; }
	bool operator==(const CirculationKeyData &rhs);
	bool operator!=(const CirculationKeyData &rhs);
	std::string toString();
	int solutionId;
	CirculationId circulationId;
	float value;
};

class CuComparatorResult
{
	friend CuSolutionComparator;
	friend void startSolutionComparatorKernel(int populationSize, CuEvaluationResult *evaluationResults, CuComparatorResult *results);

public:
	CuComparatorResult() : _devicePtr(0) { initialize(); }
	CuComparatorResult(const CuComparatorResult &other);
	~CuComparatorResult();

	CuComparatorResult& operator=(const CuComparatorResult &rhs);

	bool operator==(const CuComparatorResult &rhs);
	bool operator!=(const CuComparatorResult &rhs);

	CuComparatorResult* getDevPtr();
	void copyToHost();
	void copyToDevice();

	void initialize();

	/// <summary>
	/// Liefert die Daten der Lösung mit den niedrigsten Gesamtkosten aus der letzten Runde.
	/// </summary>
	CU_HSTDEV SolutionKeyData getLowestTotalCost() const { return _lowestTotalCost; }

	/// <summary>
	/// Liefert die Daten der Lösung mit den höchsten Gesamtkosten aus der letzten Runde.
	/// </summary>
	CU_HSTDEV SolutionKeyData getHighestTotalCost() const { return  _highestTotalCost; }

	/// <summary>
	/// Liefert die Daten der Lösung mit der niedrigsten Anzahl eingesetzter Fahrzeuge aus der letzten Runde.
	/// </summary>
	CU_HSTDEV SolutionKeyData getLowestNumOfVehicles() const { return  _lowestNumOfVehicles; }

	/// <summary>
	/// Liefert die Daten der Lösung mit der höchsten Anzahl eingesetzter Fahrzeuge aus der letzten Runde.
	/// </summary>
	CU_HSTDEV SolutionKeyData getHighestNumOfVehicles() const { return  _highestNumOfVehicles; }

	/// <summary>
	/// Liefert die Daten des Umlaufs mit dem niedrigsten Brutto-/Netto-Kostenverhältnis aus der letzten Runde.
	/// </summary>
	CU_HSTDEV CirculationKeyData getLowestCircCostRatio() const { return _lowestCircCostRatio; }

	/// <summary>
	/// Liefert die Daten des Umlaufs mit dem höchsten Brutto-/Netto-Kostenverhältnis aus der letzten Runde.
	/// </summary>
	CU_HSTDEV CirculationKeyData getHighestCircCostRatio() const { return _highestCircCostRatio; }

	void dump();

private:

	/// <summary>
	/// Enthält die Daten der Lösung mit den niedrigsten Gesamtkosten aus der letzten Runde.
	/// </summary>
	SolutionKeyData _lowestTotalCost;

	/// <summary>
	/// Enthält die Daten der Lösung mit den höchsten Gesamtkosten aus der letzten Runde.
	/// </summary>
	SolutionKeyData _highestTotalCost;

	/// <summary>
	/// Enthält die Daten der Lösung mit der niedrigsten Anzahl eingesetzter Fahrzeuge aus der letzten Runde.
	/// </summary>
	SolutionKeyData _lowestNumOfVehicles;

	/// <summary>
	/// Enthält die Daten der Lösung mit der höchsten Anzahl eingesetzter Fahrzeuge aus der letzten Runde.
	/// </summary>
	SolutionKeyData _highestNumOfVehicles;

	/// <summary>
	/// Enthält die Daten der Lösung aus der letzten Runde, die den Umlauf mit dem niedrigsten Brutto-/Netto-Kostenverhältnis beinhaltet.
	/// </summary>
	CirculationKeyData _lowestCircCostRatio;

	/// <summary>
	/// Enthält die Daten der Lösung aus der letzten Runde, die den Umlauf mit dem höchsten Brutto-/Netto-Kostenverhältnis beinhaltet.
	/// </summary>
	CirculationKeyData _highestCircCostRatio;

	CuComparatorResult* _devicePtr;
};

