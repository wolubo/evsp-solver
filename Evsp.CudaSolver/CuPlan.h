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
/// Enth�lt die Daten einer L�sung. 
/// </summary>			
class CuPlan {
public:
	CuPlan() = delete;
	CuPlan(int maxNumOfCirculations, int maxNumOfNodes);
	CuPlan(const CuPlan &other);
	~CuPlan();

	/// <summary>
	/// Reinitialisiert die L�sung (setzt die Anzahl der Uml�ufe auf 0).
	/// </summary>			
	CU_HSTDEV void reInit();

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory). Alle weiteren Aufrufe 
	/// liefern einen Pointer auf diese Kopie.
	/// </summary>			
	/// <returns>Pointer auf das Device-Objekt.</returns>
	CuPlan* getDevPtr();

	/// <summary>
	/// �berschreibt die Daten des Objekts mit denen des Objekts im Speicher der GPU (Device-Memory).
	/// </summary>			
	void copyToHost();

	void copyToDevice();

	/// <summary>
	/// F�gt der L�sung einen neuen Umlauf hinzu.
	/// </summary>			
	/// <returns>Id des neuen Umlaufs.</returns>
	CU_HSTDEV CirculationId addNewCirculation(StopId startDepot, VehicleTypeId vehicleType);

	/// <summary>
	/// H�ngt eine Aktion an einen bestehenden Umlauf an.
	/// </summary>			
	/// <param name="circulationId"></param>
	/// <param name="newNode"></param>
	/// <param name="selectedEdge"></param>
	/// <returns>Index der neuen Aktion.</returns>
	CU_HSTDEV CircStepIndex appendNode(CirculationId circulationId, NodeId newNode, EdgeId selectedEdge);

	/// <summary>
	/// L�scht die letzte dem Entscheidungspfad hinzugef�gte Entscheidung wieder. Macht sie also r�ckg�ngig.
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
	/// Enh�lt die Aktionen s�mtlicher L�sungen.
	/// row: Index der Aktion innerhalb des Umlaufs
	/// col: Id des Umlaufs
	/// </summary>			
	CuMatrix1<NodeId> *_nodes;

	// TODO _edges aus CuPlan entfernen! Redundante Information, da Kante aus n�chster Aktion rekonstruiert werden kann.
	/// <summary>
	/// Enth�lt die ausgew�hlten Kanten.
	/// row: Index der Aktion innerhalb des Umlaufs
	/// col: Id des Umlaufs
	/// </summary>			
	CuMatrix1<EdgeId> *_edges;

	/// <summary>
	/// Maximale m�gliche Anzahl von Uml�ufen.
	/// </summary>			
	int _maxNumOfCirculations;

	/// <summary>
	/// Maximal m�gliche Anzahl von Aktionen je Umlauf.
	/// </summary>			
	int _maxNumOfNodes;

	/// <summary>
	/// Aktuelle Anzahl von Uml�ufen.
	/// </summary>			
	int _numOfCirculations;

	/// <summary>
	/// Enth�lt f�r jeden Umlauf die aktuelle Anzahl von Aktionen.
	/// row: Id des Umlaufs
	/// col: Id der L�sung
	/// </summary>			
	CuVector1<int> *_numOfNodes;

	/// <summary>
	/// Enth�lt die Haltestellen-Id des Startdepots.
	/// </summary>			
	CuVector1<StopId> *_startDepot;

	/// <summary>
	/// Enth�lt die Id des dem Umlauf zugeordneten Fahrzeugtypen.
	/// </summary>			
	CuVector1<VehicleTypeId> *_vehicleType;

	/// <summary>
	/// 
	/// </summary>			
	CuPlan *_deviceObject;
};


