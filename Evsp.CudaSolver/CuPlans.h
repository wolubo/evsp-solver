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
/// Enth�lt die Entscheidungspfade aller L�sungen. Wird von CuSolverAcoGpu::run() aufgebaut.
/// Jeder Entscheidungspfad enth�lt alle Uml�ufe einer L�sung. Er beginnt im Root-Knoten (Id: 0).
/// Jeder einzelne Umlauf einer L�sung beginnt wiederum im Root-Knoten.
/// </summary>			
class CuPlans {
public:
	CuPlans() = delete;
	CuPlans(int population, int maxNumOfCirculations, int maxNumOfNodes);
	CuPlans(const CuPlans &other);
	~CuPlans();

	/// <summary>
	/// Reinitialisiert eine L�sung (setzt die Anzahl der Uml�ufe auf 0).
	/// </summary>			
	CU_HSTDEV void reInit(int solutionId);

	/// <summary>
	/// Erstellt beim ersten Aufruf eine Kopie des Objekts im Speicher der GPU (Device-Memory). Alle weiteren Aufrufe 
	/// liefern einen Pointer auf diese Kopie.
	/// </summary>			
	/// <returns>Pointer auf das Device-Objekt.</returns>
	CuPlans* getDevPtr();

	/// <summary>
	/// �berschreibt die Daten des Objekts mit denen des Objekts im Speicher der GPU (Device-Memory).
	/// </summary>			
	void copyToHost();

	void copyToDevice();

	/// <summary>
	/// F�gt einer L�sung einen neuen Umlauf hinzu.
	/// </summary>			
	/// <param name="solutionId"></param>
	/// <param name="XXX"></param>
	/// <returns>Id des neuen Umlaufs.</returns>
	CU_HSTDEV CirculationId addNewCirculation(int solutionId, StopId startDepot, VehicleTypeId vehicleType);

	/// <summary>
	/// H�ngt eine Aktion an einen bestehenden Umlauf an.
	/// </summary>			
	/// <param name="solutionId"></param>
	/// <param name="circulationId"></param>
	/// <param name="newNode"></param>
	/// <param name="selectedEdge"></param>
	/// <returns>Index der neuen Aktion.</returns>
	CU_HSTDEV CircStepIndex appendNode(int solutionId, CirculationId circulationId, NodeId newNode, EdgeId selectedEdge);

	/// <summary>
	/// L�scht die letzte dem Entscheidungspfad hinzugef�gte Entscheidung wieder. Macht sie also r�ckg�ngig.
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
	/// Enh�lt die Aktionen s�mtlicher L�sungen.
	/// x: Index der Aktion innerhalb des Umlaufs
	/// y: Id des Umlaufs
	/// z: Id der L�sung
	/// </summary>			
	Matrix3d<NodeId> *_nodes;

	// TODO _edges aus CuPlans entfernen! Redundante Information, da Kante aus n�chster Aktion rekonstruiert werden kann.
	/// <summary>
	/// Enth�lt die ausgew�hlten Kanten s�mtlicher L�sungen.
	/// x: Index der Aktion innerhalb des Umlaufs
	/// y: Id des Umlaufs
	/// z: Id der L�sung
	/// </summary>			
	Matrix3d<EdgeId> *_edges;

	/// <summary>
	/// Anzahl der betrachteten L�sungen (Gr�sse der Population).
	/// </summary>			
	int _population;

	/// <summary>
	/// Maximale m�gliche Anzahl von Uml�ufen je L�sung.
	/// </summary>			
	int _maxNumOfCirculations;

	/// <summary>
	/// Maximal m�gliche Anzahl von Aktionen je Umlauf.
	/// </summary>			
	int _maxNumOfNodes;

	/// <summary>
	/// Enth�lt f�r jede L�sung die aktuelle Anzahl von Uml�ufen.
	/// </summary>			
	CuVector1<int> *_numOfCirculations;

	/// <summary>
	/// Enth�lt f�r jeden Umlauf die Haltestellen-Id des Startdepots.
	/// </summary>			
	CuMatrix1<StopId> *_startDepot;

	/// <summary>
	/// Enth�lt f�r jeden Umlauf die Id des zugeordneten Fahrzeugtypen.
	/// </summary>			
	CuMatrix1<VehicleTypeId> *_vehicleType;

	/// <summary>
	/// Enth�lt f�r jeden Umlauf die aktuelle Anzahl von Aktionen.
	/// row: Id des Umlaufs
	/// col: Id der L�sung
	/// </summary>			
	CuMatrix1<int> *_numOfNodes;

	/// <summary>
	/// 
	/// </summary>			
	CuPlans *_deviceObject;
};


