#pragma once

#include "CuEdges.h"
#include "CuNodes.h"
#include "CuProblem.h"
#include "MatrixCreator.h"


/// <summary>
/// "Public Transport Network"
/// Repräsentiert das Entscheidungsnetz für die ACO.
/// </summary>
class CuConstructionGraph {
public:

	/// <summary>
	/// 
	/// </summary>
	CuEdges *edges;

	/// <summary>
	/// 
	/// </summary>
	CuNodes nodes;

	/// <summary>
	/// Konstruktor.
	/// Erzeugt das Netz mithilfe eines Problem-Objekts.
	/// </summary>
	/// <param name="problem"></param>
	/// <param name="initialWeight"></param>
	/// <param name="connectionMatrix"></param>
	CuConstructionGraph(std::shared_ptr<CuProblem> problem, float initialWeight, PlattformConfig plattform, bool performCheck);

	~CuConstructionGraph();

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory) und speichert die Adresse der
	/// Kopie im Attribut 'deviceObject'. Alle weiteren Aufrufe liefern einen Pointer auf diese Kopie (also den Inhalt von
	/// 'deviceObject').
	/// </summary>
	/// <returns></returns>
	CuConstructionGraph* getDevPtr();

	/// <summary>
	/// 
	/// </summary>
	void copyToHost();

	/// <summary>
	/// Gibt die Anzahl der Knoten und die Anzahl der Kanten aus.
	/// </summary>
	void printStatistic();

	/// <summary>
	/// Gibt die alle Knoten und alle Kanten auf der Konsole aus.
	/// </summary>
	void dumpDecisionNet(std::shared_ptr<CuProblem> problem);

private:

	/// <summary>
	/// Erzeugt alle Knoten. Füllt also das Attribut 'nodes'.
	/// </summary>
	void createNodes(std::shared_ptr<CuProblem> problem);

	/// <summary>
	/// Erzeugt alle Kanten. Füllt also das Attribut 'edges'.
	/// </summary>
	void createEdges(std::shared_ptr<CuProblem> problem, float initialWeight, PlattformConfig plattform);

	void check(std::shared_ptr<CuProblem> problem);

	/// <summary>
	/// 
	/// </summary>
	CuConstructionGraph *_deviceObject;
};

