#pragma once
#pragma warning(disable: 4244)

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuPlan.h"
#include "CuVector1.hpp"
#include "CuMatrix1.hpp"
#include "EVSP.BaseClasses/PointInTime.hpp"


class CuEvaluationResult
{
public:
	CuEvaluationResult(int numOfSolutions, int maxNumOfCirculations);
	CuEvaluationResult(const CuEvaluationResult& other);
	~CuEvaluationResult();

	CuEvaluationResult& operator=(const CuEvaluationResult &rhs) { assert(false); return *this; }

	bool operator==(const CuEvaluationResult &rhs);
	bool operator!=(const CuEvaluationResult &rhs);

	CU_HSTDEV void initialize();
	CU_HSTDEV void initialize(int solutionId);

	CuEvaluationResult* getDevPtr();
	void copyToHost();
	void copyToDevice();

	CU_HSTDEV int getNumOfSolutions() const { return _numOfSolutions; }

	CU_HSTDEV void addCirculation(int solutionId) {
		((*_numOfCirculations)[solutionId])++;
	}

	CU_HSTDEV int getNumOfCirculations(int solutionId) {
		return (*_numOfCirculations)[solutionId];
	}

	CU_HSTDEV AmountOfMoney getTotalCost(int solutionId) const { return (*_totalCost)[solutionId]; }

	CU_HSTDEV void setTotalCost(int solutionId, AmountOfMoney value) {
		(*_totalCost)[solutionId] = value; 
	}

	CU_HSTDEV void addToTotalCost(int solutionId, AmountOfMoney value) {
		(*_totalCost)[solutionId] = (*_totalCost)[solutionId] + value; 
	}

	CU_HSTDEV PointInTime getCircStartTime(int solutionId, CirculationId circulationId) { return _circStartTime->get(solutionId, (short)circulationId); }

	CU_HSTDEV void setCircStartTime(int solutionId, CirculationId circulationId, PointInTime startTime) {
		_circStartTime->set(solutionId, (short)circulationId, startTime);
	}

	/// <summary>
	/// Liefert die Gesamtkosten des Umlaufs einschließlich der Anschaffungskosten für das eingesetzte Fahrzeug.
	/// </summary>
	CU_HSTDEV AmountOfMoney getCircCost(int solutionId, CirculationId circulationId) { return _circCost->get(solutionId, (short)circulationId); }

	/// <summary>
	/// Setzt die Gesamtkosten des Umlaufs einschließlich der Anschaffungskosten für das eingesetzte Fahrzeug.
	/// </summary>
	CU_HSTDEV void setCircCost(int solutionId, CirculationId circulationId, AmountOfMoney grossCost) {
		_circCost->set(solutionId, (short)circulationId, grossCost);
	}

	/// <summary>
	/// Setzt das Brutto-/Netto-Kostenverhältnis eines Umlaufs. 
	/// Das Kostenverhältnis eines Umlaufs gibt das Verhältnis der Durchführungskosten (inkl. aller anderen Kosten wie Wartezeiten, 
	/// Fahrzeuganschaffung, Aufladungen und Leerfahrten) zu den Kosten, die für die reine Durchführung der Servicefahrten entstehen 
	/// würden wieder. Letztere sind ein Idealwert, der in der Praxis nicht erreichbar sein wird, weil ja zumindest für die 
	/// Anschaffung des Fahrzeugs und für unvermeidliche Wartezeiten Kosten anfallen.
	/// Der beste erreichbare Wert ist 1.0. Höhere Werte bedeuten eine schlechtere Effizienz des betreffenden Umlaufs (sub-optimaler 
	/// Einsatz der Betriebsmittel).
	/// </summary>
	CU_HSTDEV void setCircCostRatio(int solutionId, CirculationId circulationId, float ratio) {
		assert(ratio >= 1.0f);
		_circCostRatio->set(solutionId, (short)circulationId, ratio);
	}

	/// <summary>
	/// Liefert das Brutto-/Netto-Kostenverhältnis eines Umlaufs. 
	/// Das Kostenverhältnis eines Umlaufs gibt das Verhältnis der Durchführungskosten (inkl. aller anderen Kosten wie Wartezeiten, 
	/// Fahrzeuganschaffung, Aufladungen und Leerfahrten) zu den Kosten, die für die reine Durchführung der Servicefahrten entstehen 
	/// würden wieder. Letztere sind ein Idealwert, der in der Praxis nicht erreichbar sein wird, weil ja zumindest für die 
	/// Anschaffung des Fahrzeugs und für unvermeidliche Wartezeiten Kosten anfallen.
	/// Der beste erreichbare Wert ist 1.0. Höhere Werte bedeuten eine schlechtere Effizienz des betreffenden Umlaufs (sub-optimaler 
	/// Einsatz der Betriebsmittel).
	/// </summary>
	CU_HSTDEV float getCircCostRatio(int solutionId, CirculationId circulationId) {
		return _circCostRatio->get(solutionId, (short)circulationId);
	}

	void dump();

private:
	CuEvaluationResult() {}
	int _numOfSolutions;
	CuVector1<int> *_numOfCirculations;
	CuVector1<AmountOfMoney> *_totalCost;
	CuMatrix1<PointInTime> *_circStartTime;
	CuMatrix1<AmountOfMoney> *_circCost;
	CuMatrix1<float> *_circCostRatio;
	CuEvaluationResult *_devicePtr;
};

