#pragma once

#include "device_launch_parameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuPlans.h"
#include "CuEvaluationResult.h"
#include "CuComparatorResult.h"

void startSolutionComparatorKernel(int populationSize, CuEvaluationResult *evaluationResults, CuComparatorResult *results);
