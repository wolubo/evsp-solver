#pragma once

#include <memory>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EvspLimits.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuMatrix1.hpp"
#include "CuVector1.hpp"
#include "CuEdges.h"
#include "Matrix3d.hpp"

using namespace std;

class CuPlan;
class CuConstructionGraph;

/// <summary>
/// Enthält die Entscheidungspfade aller Lösungen. Wird von CuSolverAcoGpu::run() aufgebaut.
/// Jeder Entscheidungspfad enthält alle Umläufe einer Lösung. Er beginnt im Root-Knoten (Id: 0).
/// Jeder einzelne Umlauf einer Lösung beginnt wiederum im Root-Knoten.
/// </summary>			
class CuPlans {
public:
	CuPlans() = delete;
	CuPlans(int population, int maxNumOfCirculations, int maxNumOfNodes);
	CuPlans(const CuPlans &other);
	~CuPlans();

	/// <summary>
	/// Reinitialisiert eine Lösung (setzt die Anzahl der Umläufe auf 0).
	/// </summary>			
	CU_HSTDEV void reInit(int solutionId);

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory). Alle weiteren Aufrufe 
	/// liefern einen Pointer auf diese Kopie.
	/// </summary>			
	/// <returns>Pointer auf das Device-Objekt.</returns>
	CuPlans* getDevPtr();

	/// <summary>
	/// Überschreibt die Daten des Objekts mit denen des Objekts im Speicher der GPU (Device-Memory).
	/// </summary>			
	void copyToHost();

	void copyToDevice();

	/// <summary>
	/// Fügt einer Lösung einen neuen Umlauf hinzu.
	/// </summary>			
	/// <param name="solutionId"></param>
	/// <param name="XXX"></param>
	/// <returns>Id des neuen Umlaufs.</returns>
	CU_HSTDEV CirculationId addNewCirculation(int solutionId, StopId startDepot, VehicleTypeId vehicleType);

	/// <summary>
	/// Hängt eine Aktion an einen bestehenden Umlauf an.
	/// </summary>			
	/// <param name="solutionId"></param>
	/// <param name="circulationId"></param>
	/// <param name="newNode"></param>
	/// <param name="selectedEdge"></param>
	/// <returns>Index der neuen Aktion.</returns>
	CU_HSTDEV CircStepIndex appendNode(int solutionId, CirculationId circulationId, NodeId newNode, EdgeId selectedEdge);

	/// <summary>
	/// Löscht die letzte dem Entscheidungspfad hinzugefügte Entscheidung wieder. Macht sie also rückgängig.
	/// </summary>			
	/// <param name="solutionId"></param>
	/// <param name="circulationId"></param>
	//CU_HSTDEV void revertLastDecision(int solutionId, CirculationId circulationId);

	/// <summary>
	/// Liefert die Id einer Aktion.
	/// </summary>			
	/// <param name="solutionId"></param>
	/// <param name="circulationId"></param>
	/// <param name="stepIndex"></param>
	/// <returns></returns>
	CU_HSTDEV NodeId getNodeId(int solutionId, CirculationId circulationId, CircStepIndex stepIndex);

	/// <summary>
	/// Liefert die Id einer Kante.
	/// </summary>			
	/// <param name="solutionId"></param>
	/// <param name="circulationId"></param>
	/// <param name="stepIndex"></param>
	/// <returns></returns>
	CU_HSTDEV EdgeId getEdgeId(int solutionId, CirculationId circulationId, CircStepIndex stepIndex);

	CU_HSTDEV int getNumOfSolutions() const { return _population; }
	CU_HSTDEV int getNumOfCirculations(int solutionId) const;
	CU_HSTDEV int getNumOfNodes(int solutionId, CirculationId circulationId) const;

	CU_HSTDEV int getMaxNumOfCirculations() const { return _maxNumOfCirculations; }
	CU_HSTDEV int getMaxNumOfNodes() const { return _maxNumOfNodes; }

	CU_HSTDEV StopId getDepotId(int solutionId, CirculationId circulationId) const;
	CU_HSTDEV void setDepotId(int solutionId, CirculationId circulationId, StopId depotId);

	CU_HSTDEV VehicleTypeId getVehicleTypeId(int solutionId, CirculationId circulationId) const;
	CU_HSTDEV void setVehicleTypeId(int solutionId, CirculationId circulationId, VehicleTypeId vehicleTypeId);

	std::shared_ptr<CuPlan> getPlan(int solutionId);
	CU_HSTDEV CuPlan& getPlan(int solutionId, CuPlan &result);

	void dump(shared_ptr<CuConstructionGraph> ptn, int solutionId);
	CU_HSTDEV void dump(CuConstructionGraph *ptn, int solutionId);

private:
	/// <summary>
	/// Enhält die Aktionen sämtlicher Lösungen.
	/// x: Index der Aktion innerhalb des Umlaufs
	/// y: Id des Umlaufs
	/// z: Id der Lösung
	/// </summary>			
	Matrix3d<NodeId> *_nodes;

	// TODO _edges aus CuPlans entfernen! Redundante Information, da Kante aus nächster Aktion rekonstruiert werden kann.
	/// <summary>
	/// Enthält die ausgewählten Kanten sämtlicher Lösungen.
	/// x: Index der Aktion innerhalb des Umlaufs
	/// y: Id des Umlaufs
	/// z: Id der Lösung
	/// </summary>			
	Matrix3d<EdgeId> *_edges;

	/// <summary>
	/// Anzahl der betrachteten Lösungen (Grösse der Population).
	/// </summary>			
	int _population;

	/// <summary>
	/// Maximale mögliche Anzahl von Umläufen je Lösung.
	/// </summary>			
	int _maxNumOfCirculations;

	/// <summary>
	/// Maximal mögliche Anzahl von Aktionen je Umlauf.
	/// </summary>			
	int _maxNumOfNodes;

	/// <summary>
	/// Enthält für jede Lösung die aktuelle Anzahl von Umläufen.
	/// </summary>			
	CuVector1<int> *_numOfCirculations;

	/// <summary>
	/// Enthält für jeden Umlauf die Haltestellen-Id des Startdepots.
	/// </summary>			
	CuMatrix1<StopId> *_startDepot;

	/// <summary>
	/// Enthält für jeden Umlauf die Id des zugeordneten Fahrzeugtypen.
	/// </summary>			
	CuMatrix1<VehicleTypeId> *_vehicleType;

	/// <summary>
	/// Enthält für jeden Umlauf die aktuelle Anzahl von Aktionen.
	/// row: Id des Umlaufs
	/// col: Id der Lösung
	/// </summary>			
	CuMatrix1<int> *_numOfNodes;

	/// <summary>
	/// 
	/// </summary>			
	CuPlans *_deviceObject;
};


