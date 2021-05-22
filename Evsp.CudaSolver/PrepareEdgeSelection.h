#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuMatrix1.hpp"
#include "CuLockVector1.h"
#include "CuConstructionGraph.h"
#include "CuEdges.h"
#include "CuNodes.h"
#include "CuProblem.h"
#include "CuPlans.h"
#include "MatrixCreator.h"
#include "CuVector2.hpp"
#include "CuVector1.hpp"
#include "AntsRaceState.h"

/// <summary>
/// Sammle alle aktiven Kanten des aktuellen Knotens (auf der CPU).
/// Eine Kante ist aktiv, wenn sie auf einen aktiven Knoten verweist.
/// Kanten zu Servicefahrten werden nur dann als aktiv angesehen, wenn die Servicefahrt mit dem aktuellen Fahrzeug
/// durchführbar ist.
/// Kanten zu Ladestationen werden nur dann als aktiv angesehen, wenn eine gewisse Batteriekapazität unterschritten ist.
/// Falls die Mindestbatteriekapazität unterschritten ist gelten nur noch Kanten zum Startdepot und zu Ladestationen als aktiv.
/// </summary>
CU_HSTDEV int prepareEdgeSelection(CuVector1<EdgeId> *activeEdges, CuConstructionGraph *ptn, NodeId currNodeId, CuVector1<bool> *activityStateOfNodes, AntsRaceState *state, CuVector1<float> *weights, float &totalWeight);


/// <summary>
/// Sammle alle aktiven Kanten des aktuellen Knotens (auf der GPU).
/// Eine Kante ist aktiv, wenn sie auf einen aktiven Knoten verweist.
/// Kanten zu Servicefahrten werden nur dann als aktiv angesehen, wenn die Servicefahrt mit dem aktuellen Fahrzeug
/// durchführbar ist.
/// Kanten zu Ladestationen werden nur dann als aktiv angesehen, wenn eine gewisse Batteriekapazität unterschritten ist.
/// Falls die Mindestbatteriekapazität unterschritten ist gelten nur noch Kanten zum Startdepot und zu Ladestationen als aktiv.
/// </summary>
CU_DEV int prepareEdgeSelectionGpu(CuVector1<EdgeId> *activeEdges, CuConstructionGraph *ptn, NodeId currNodeId, CuVector1<bool> *activityStateOfNodes, AntsRaceState *state, CuVector1<float> *weights, float &totalWeight, CuLockVector1 *nodeLock);
