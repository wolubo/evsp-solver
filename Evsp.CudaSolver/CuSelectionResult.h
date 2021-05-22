#pragma once

#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuPlans.h"
#include "CuPlan.h"
#include "Temperature.hpp"
#include "CuVector1.hpp"
#include "CuComparatorResult.h"

//class CuSolutionSelectorAco;

class CuSelectionResult
{
	//friend CuSolutionSelectorAco;
public:
	CuSelectionResult(int populationSize, int maxNumOfCirculations, int maxNumOfNodes);
	~CuSelectionResult();

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory). Alle weiteren Aufrufe 
	/// liefern einen Pointer auf diese Kopie.
	/// </summary>			
	/// <returns>Pointer auf das Device-Objekt.</returns>
	CuSelectionResult* getDevPtr();

	/// <summary>
	/// Überschreibt die Daten des Objekts mit denen des Objekts im Speicher der GPU (Device-Memory).
	/// </summary>			
	void copyToHost();

	void copyToDevice();

	/// <summary>
	/// True: Es existiert eine neue beste Lösung.
	/// False: Es konnte in der letzten Runde keine neue beste Lösung gefunden werden.
	/// </summary>
	CU_HSTDEV bool hasNewBestSolution() const { return _hasNewBestSolution; }

	/// <summary>
	/// Prüft, ob eine neue beste Lösung gefunden wurde. Wenn dem so ist werden deren Daten gespeichert und die neue
	/// beste Lösung wird als aktiv markiert.
	/// </summary>
	/// <param name="lowestTotalCost"></param>
	/// <returns>Liefert TRUE, wenn die beste bisherige Lösung durch die neue Lösung ersetzt wurde.</returns>
	CU_HSTDEV bool checkTheBestSolutionInClass(SolutionKeyData toBeChecked);
	
	/// <summary>
	/// Prüft, ob schlechteste Lösung gefunden wurde und speichert die Daten ggf.
	/// </summary>
	/// <param name="highestTotalCost"></param>
	/// <returns>Liefert TRUE, wenn die beste bisherige Lösung durch die neue Lösung ersetzt wurde.</returns>
	CU_HSTDEV bool checkTheWorstSolutionInClass(SolutionKeyData toBeChecked);

	/// <summary>
	/// Liefert die Daten der besten gefundenen Lösung.
	/// </summary>
	CU_HSTDEV SolutionKeyData getBestSolution() const { return _bestSolutionKeyData; }

	/// <summary>
	/// Liefert die Daten der schlechtesten gefundenen Lösung.
	/// </summary>
	CU_HSTDEV SolutionKeyData getWorstSolution() const { return _worstSolutionKeyData; }

	CU_HSTDEV bool isAlive(int solutionId) const { return _alive->get(solutionId); }
	CU_HSTDEV void markAsAlive(int solutionId) { _alive->set(solutionId, true); }
	CU_HSTDEV void markAllAsAlive() { _alive->setAll(true); }
	CU_HSTDEV void markAsDead(int solutionId) { (*_alive)[solutionId] = false; }
	CU_HSTDEV void markAllAsDead() { _alive->setAll(false); }

private:

	int _populationSize;

	/// <summary>
	/// True: Es existiert eine neue beste Lösung.
	/// False: Es konnte in der letzten Runde keine neue beste Lösung gefunden werden.
	/// </summary>
	bool _hasNewBestSolution;

	/// <summary>
	/// Enthält die Daten der besten gefundenen Lösung.
	/// </summary>
	SolutionKeyData _bestSolutionKeyData;

	/// <summary>
	/// Enthält die Daten der schlechtesten gefundenen Lösung.
	/// </summary>
	SolutionKeyData _worstSolutionKeyData;

	/// <summary>
	/// Jede Lösung kann entweder aktiv (true, "alive") oder inaktiv (false, "dead") sein.
	/// </summary>			
	CuVector1<bool> *_alive;

	/// <summary>
	/// 
	/// </summary>			
	CuSelectionResult *_devicePtr;
};

