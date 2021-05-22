#pragma once

#include <memory>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "EvspLimits.h"
#include "CuMatrix1.hpp"
#include "CuVector1.hpp"
#include "CuEdges.h"
#include "Matrix3d.hpp"

using namespace std;

class CuConstructionGraph;

/// <summary>
/// Enthält die Daten einer Lösung. 
/// </summary>			
class CuPlan {
public:
	CuPlan() = delete;
	CuPlan(int maxNumOfCirculations, int maxNumOfNodes);
	CuPlan(const CuPlan &other);
	~CuPlan();

	/// <summary>
	/// Reinitialisiert die Lösung (setzt die Anzahl der Umläufe auf 0).
	/// </summary>			
	CU_HSTDEV void reInit();

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory). Alle weiteren Aufrufe 
	/// liefern einen Pointer auf diese Kopie.
	/// </summary>			
	/// <returns>Pointer auf das Device-Objekt.</returns>
	CuPlan* getDevPtr();

	/// <summary>
	/// Überschreibt die Daten des Objekts mit denen des Objekts im Speicher der GPU (Device-Memory).
	/// </summary>			
	void copyToHost();

	void copyToDevice();

	/// <summary>
	/// Fügt der Lösung einen neuen Umlauf hinzu.
	/// </summary>			
	/// <returns>Id des neuen Umlaufs.</returns>
	CU_HSTDEV CirculationId addNewCirculation(StopId startDepot, VehicleTypeId vehicleType);

	/// <summary>
	/// Hängt eine Aktion an einen bestehenden Umlauf an.
	/// </summary>			
	/// <param name="circulationId"></param>
	/// <param name="newNode"></param>
	/// <param name="selectedEdge"></param>
	/// <returns>Index der neuen Aktion.</returns>
	CU_HSTDEV CircStepIndex appendNode(CirculationId circulationId, NodeId newNode, EdgeId selectedEdge);

	/// <summary>
	/// Löscht die letzte dem Entscheidungspfad hinzugefügte Entscheidung wieder. Macht sie also rückgängig.
	/// </summary>			
	/// <param name="circulationId"></param>
	//CU_HSTDEV void revertLastDecision(CirculationId circulationId);

	/// <summary>
	/// Liefert die Id einer Aktion.
	/// </summary>			
	/// <param name="circulationId"></param>
	/// <param name="stepIndex"></param>
	/// <returns></returns>
	CU_HSTDEV NodeId getNodeId(CirculationId circulationId, CircStepIndex stepIndex);

	/// <summary>
	/// Liefert die Id einer Kante.
	/// </summary>			
	/// <param name="circulationId"></param>
	/// <param name="stepIndex"></param>
	/// <returns></returns>
	CU_HSTDEV EdgeId getEdgeId(CirculationId circulationId, CircStepIndex stepIndex);

	CU_HSTDEV int getNumOfCirculations() const { return _numOfCirculations; }
	CU_HSTDEV int getNumOfNodes(CirculationId circulationId);
	CU_HSTDEV int getMaxNumOfCirculations() { return _maxNumOfCirculations; }
	CU_HSTDEV int getMaxNumOfNodes() { return _maxNumOfNodes; }

	void dump(shared_ptr<CuConstructionGraph> ptn);

private:
	/// <summary>
	/// Enhält die Aktionen sämtlicher Lösungen.
	/// row: Index der Aktion innerhalb des Umlaufs
	/// col: Id des Umlaufs
	/// </summary>			
	CuMatrix1<NodeId> *_nodes;

	// TODO _edges aus CuPlan entfernen! Redundante Information, da Kante aus nächster Aktion rekonstruiert werden kann.
	/// <summary>
	/// Enthält die ausgewählten Kanten.
	/// row: Index der Aktion innerhalb des Umlaufs
	/// col: Id des Umlaufs
	/// </summary>			
	CuMatrix1<EdgeId> *_edges;

	/// <summary>
	/// Maximale mögliche Anzahl von Umläufen.
	/// </summary>			
	int _maxNumOfCirculations;

	/// <summary>
	/// Maximal mögliche Anzahl von Aktionen je Umlauf.
	/// </summary>			
	int _maxNumOfNodes;

	/// <summary>
	/// Aktuelle Anzahl von Umläufen.
	/// </summary>			
	int _numOfCirculations;

	/// <summary>
	/// Enthält für jeden Umlauf die aktuelle Anzahl von Aktionen.
	/// row: Id des Umlaufs
	/// col: Id der Lösung
	/// </summary>			
	CuVector1<int> *_numOfNodes;

	/// <summary>
	/// Enthält die Haltestellen-Id des Startdepots.
	/// </summary>			
	CuVector1<StopId> *_startDepot;

	/// <summary>
	/// Enthält die Id des dem Umlauf zugeordneten Fahrzeugtypen.
	/// </summary>			
	CuVector1<VehicleTypeId> *_vehicleType;

	/// <summary>
	/// 
	/// </summary>			
	CuPlan *_deviceObject;
};


