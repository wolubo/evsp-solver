#pragma once

#include "EVSP.BaseClasses/Typedefs.h"
#include "CuConstructionGraph.h"
#include "CuProblem.h"
#include "MatrixCreator.h"

void createPtnEdgesOnCpu(CuConstructionGraph *ptn, std::shared_ptr<CuProblem> problem, shared_ptr<VehicleTypeGroupIntersection> vtgIntersect, float initialWeight, int numberOfNodes);
