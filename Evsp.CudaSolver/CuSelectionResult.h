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
	/// �berschreibt die Daten des Objekts mit denen des Objekts im Speicher der GPU (Device-Memory).
	/// </summary>			
	void copyToHost();

	void copyToDevice();

	/// <summary>
	/// True: Es existiert eine neue beste L�sung.
	/// False: Es konnte in der letzten Runde keine neue beste L�sung gefunden werden.
	/// </summary>
	CU_HSTDEV bool hasNewBestSolution() const { return _hasNewBestSolution; }

	/// <summary>
	/// Pr�ft, ob eine neue beste L�sung gefunden wurde. Wenn dem so ist werden deren Daten gespeichert und die neue
	/// beste L�sung wird als aktiv markiert.
	/// </summary>
	/// <param name="lowestTotalCost"></param>
	/// <returns>Liefert TRUE, wenn die beste bisherige L�sung durch die neue L�sung ersetzt wurde.</returns>
	CU_HSTDEV bool checkTheBestSolutionInClass(SolutionKeyData toBeChecked);
	
	/// <summary>
	/// Pr�ft, ob schlechteste L�sung gefunden wurde und speichert die Daten ggf.
	/// </summary>
	/// <param name="highestTotalCost"></param>
	/// <returns>Liefert TRUE, wenn die beste bisherige L�sung durch die neue L�sung ersetzt wurde.</returns>
	CU_HSTDEV bool checkTheWorstSolutionInClass(SolutionKeyData toBeChecked);

	/// <summary>
	/// Liefert die Daten der besten gefundenen L�sung.
	/// </summary>
	CU_HSTDEV SolutionKeyData getBestSolution() const { return _bestSolutionKeyData; }

	/// <summary>
	/// Liefert die Daten der schlechtesten gefundenen L�sung.
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
	/// True: Es existiert eine neue beste L�sung.
	/// False: Es konnte in der letzten Runde keine neue beste L�sung gefunden werden.
	/// </summary>
	bool _hasNewBestSolution;

	/// <summary>
	/// Enth�lt die Daten der besten gefundenen L�sung.
	/// </summary>
	SolutionKeyData _bestSolutionKeyData;

	/// <summary>
	/// Enth�lt die Daten der schlechtesten gefundenen L�sung.
	/// </summary>
	SolutionKeyData _worstSolutionKeyData;

	/// <summary>
	/// Jede L�sung kann entweder aktiv (true, "alive") oder inaktiv (false, "dead") sein.
	/// </summary>			
	CuVector1<bool> *_alive;

	/// <summary>
	/// 
	/// </summary>			
	CuSelectionResult *_devicePtr;
};

