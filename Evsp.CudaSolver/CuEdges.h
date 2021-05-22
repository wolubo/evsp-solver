#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EvspLimits.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuMatrix1.hpp"
#include "CuVector1.hpp"


/// <summary>
/// Verwaltet die Kanten des Netzes. Die Kanten sind logisch in einer zweidimendionalen CuMatrix2 angeordnet. Der Zeilenindex 
/// entspricht der Id des Knotens, von dem die Kante ausgeht. Der Spaltenindex entspricht der laufenden Nummer der ausgehenden 
/// Kante des Knotens.
/// </summary>
class CuEdges {
public:

	CuEdges() = delete;
	CuEdges(int numOfNodes, int maxNumOfOutEdges);
	~CuEdges();

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory) und speichert 
	/// die Adresse der Kopie im Attribut 'deviceObject'. Alle weiteren Aufrufe liefern einen Pointer auf 
	/// diese Kopie (also den Inhalt von 'deviceObject').
	/// </summary>			
	/// <returns>Pointer auf das Device-Objekt.</returns>
	CuEdges* getDevPtr();

	/// <summary>
	/// Überschreibt die Daten des Objekts mit denen des Objekts im Speicher der GPU (Device-Memory).
	/// </summary>			
	void copyToHost();

	/// <summary>
	/// Überschreibt die Daten im Device-Speicher mit denen des Objekts. Kopiert Änderungen vom Host also aufs Device.
	/// Das Objekt muss vorab bereits mit getDevPtr() in den Device-Speicher übertragen worden sein.
	/// </summary>			
	void copyToDevice();

	/// <summary>
	 /// Fügt einem Knoten eine neue Kante hinzu.
	 /// </summary>
	 /// <param name="fromNode"></param>
	 /// <param name="toNode"></param>
	 /// <param name="initWeight"></param>
	 /// <returns>Id der neuen Kante.</returns>
	CU_HSTDEV EdgeId addEdge(NodeId fromNode, NodeId toNode, float initWeight);

	/// <summary>
	/// Jeder Kante ist ein Gewicht zugeordnet, das benötigt wird, um eine gewichtete Zufallsauswahl zu treffen.
	/// </summary>
	/// <param name="fromNode"></param>
	/// <param name="edgeIndex"></param>
	/// <returns></returns>
	CU_HSTDEV float getWeight(NodeId fromNode, EdgeId edgeIndex);

	/// <summary>
	/// Aktualisiert das Kantengewicht.
	/// </summary>
	/// <param name="fromNode"></param>
	/// <param name="edgeIndex"></param>
	/// <param name="newWeight"></param>
	CU_HSTDEV void setWeight(NodeId fromNode, EdgeId edgeIndex, float newWeight);

	/// <summary>
	/// Addiert eine neue Spur auf das Kantengewicht.
	/// </summary>
	/// <param name="fromNode"></param>
	/// <param name="edgeIndex"></param>
	/// <param name="newTrack"></param>
	CU_HSTDEV void addWeight(NodeId fromNode, EdgeId edgeIndex, float newTrack);


	/// <summary>
	/// Schwächt alle bestehenden Spuren ab.
	/// </summary>
	/// <param name="fading">Faktor, mit dem alle bestehenden Spuren multipliziert werden.</param>
	CU_HSTDEV void fadeTracks(float fading);

	
	/// <summary>
	/// Dieser Zähler hält fest, wie oft die Kante bereits verwendet wurde. Dieser Wert kann bei der Neuberechnung der Kantengewichte
	/// verwendet werden.
	/// </summary>
	/// <param name="fromNode"></param>
	/// <param name="edgeIndex"></param>
	/// <returns></returns>
	CU_HSTDEV int getVisitCounter(NodeId fromNode, EdgeId edgeIndex);

	/// <summary>
	/// Erhöht den Besuchszähler.
	/// </summary>
	/// <param name="fromNode"></param>
	/// <param name="edgeIndex"></param>
	/// <param name="value"></param>
	/// <returns></returns>
	CU_HSTDEV void addToVisitCounter(NodeId fromNode, EdgeId edgeIndex, int value);

	/// <summary>
	/// Setzt den Besuchszähler wieder auf 0.
	/// </summary>
	/// <param name="fromNode"></param>
	/// <param name="edgeIndex"></param>
	/// <param name="value"></param>
	/// <returns></returns>
	CU_HSTDEV void resetVisitCounter(NodeId fromNode, EdgeId edgeIndex);

	/// <summary>
	/// Liefert die Id des Zielknotens der Kante.
	/// </summary>
	/// <param name="fromNode"></param>
	/// <param name="edgeIndex"></param>
	/// <returns></returns>
	CU_HSTDEV NodeId getTargetNode(NodeId fromNode, EdgeId edgeIndex);

	/// <summary>
	/// 
	/// </summary>
	/// <param name="fromNode"></param>
	/// <returns></returns>
	CU_HSTDEV int getNumOfOutEdges(NodeId fromNode);

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	CU_HSTDEV int getNumOfNodes() { return _numOfNodes; }

	/// <summary>
	/// 
	/// </summary>
	/// <returns></returns>
	CU_HSTDEV int getMaxNumOfOutEdges() { return _maxNumOfOutEdges; }

private:

	int _numOfNodes;

	int _maxNumOfOutEdges;

	/// <summary>
	/// Jeder Kante ist ein Gewicht zugeordnet, das benötigt wird, um eine gewichtete Zufallsauswahl zu treffen.
	/// </summary>
	CuMatrix1<float> *_weight;

	/// <summary>
	/// Dieser Zähler hält fest, wie oft die Kante bereits verwendet wurde. Dieser Wert kann bei der Neuberechnung der Kantengewichte
	/// verwendet werden.
	/// </summary>
	CuMatrix1<int> *_visitCounter;

	/// <summary>
	/// Verwaltet die Anzahl der ausgehenden Kanten der einzelnen Knoten. Der Index entspricht der Id der Knoten.
	/// Gibt also die Anzahl der Einträge des Knotens in _trgNode an.
	/// </summary>
	CuVector1<ushort> *_numOfOutEdges;

	/// <summary>
	/// Enthält die Id des Zielknotens der Kante.
	/// </summary>
	CuMatrix1<NodeId> *_trgNodeId;

	CuEdges *_devicePtr;
};

