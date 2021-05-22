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
	/// Liefert die Daten der L�sung mit den niedrigsten Gesamtkosten aus der letzten Runde.
	/// </summary>
	CU_HSTDEV SolutionKeyData getLowestTotalCost() const { return _lowestTotalCost; }

	/// <summary>
	/// Liefert die Daten der L�sung mit den h�chsten Gesamtkosten aus der letzten Runde.
	/// </summary>
	CU_HSTDEV SolutionKeyData getHighestTotalCost() const { return  _highestTotalCost; }

	/// <summary>
	/// Liefert die Daten der L�sung mit der niedrigsten Anzahl eingesetzter Fahrzeuge aus der letzten Runde.
	/// </summary>
	CU_HSTDEV SolutionKeyData getLowestNumOfVehicles() const { return  _lowestNumOfVehicles; }

	/// <summary>
	/// Liefert die Daten der L�sung mit der h�chsten Anzahl eingesetzter Fahrzeuge aus der letzten Runde.
	/// </summary>
	CU_HSTDEV SolutionKeyData getHighestNumOfVehicles() const { return  _highestNumOfVehicles; }

	/// <summary>
	/// Liefert die Daten des Umlaufs mit dem niedrigsten Brutto-/Netto-Kostenverh�ltnis aus der letzten Runde.
	/// </summary>
	CU_HSTDEV CirculationKeyData getLowestCircCostRatio() const { return _lowestCircCostRatio; }

	/// <summary>
	/// Liefert die Daten des Umlaufs mit dem h�chsten Brutto-/Netto-Kostenverh�ltnis aus der letzten Runde.
	/// </summary>
	CU_HSTDEV CirculationKeyData getHighestCircCostRatio() const { return _highestCircCostRatio; }

	void dump();

private:

	/// <summary>
	/// Enth�lt die Daten der L�sung mit den niedrigsten Gesamtkosten aus der letzten Runde.
	/// </summary>
	SolutionKeyData _lowestTotalCost;

	/// <summary>
	/// Enth�lt die Daten der L�sung mit den h�chsten Gesamtkosten aus der letzten Runde.
	/// </summary>
	SolutionKeyData _highestTotalCost;

	/// <summary>
	/// Enth�lt die Daten der L�sung mit der niedrigsten Anzahl eingesetzter Fahrzeuge aus der letzten Runde.
	/// </summary>
	SolutionKeyData _lowestNumOfVehicles;

	/// <summary>
	/// Enth�lt die Daten der L�sung mit der h�chsten Anzahl eingesetzter Fahrzeuge aus der letzten Runde.
	/// </summary>
	SolutionKeyData _highestNumOfVehicles;

	/// <summary>
	/// Enth�lt die Daten der L�sung aus der letzten Runde, die den Umlauf mit dem niedrigsten Brutto-/Netto-Kostenverh�ltnis beinhaltet.
	/// </summary>
	CirculationKeyData _lowestCircCostRatio;

	/// <summary>
	/// Enth�lt die Daten der L�sung aus der letzten Runde, die den Umlauf mit dem h�chsten Brutto-/Netto-Kostenverh�ltnis beinhaltet.
	/// </summary>
	CirculationKeyData _highestCircCostRatio;

	CuComparatorResult* _devicePtr;
};

