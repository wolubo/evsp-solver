#include "CirculationStats.h"


CirculationStats::CirculationStats()
	:totalCost(0), timeDependentCosts(0), distanceDependentCosts(0), minimumCapacity(0.0f), remainingCapacity(0.0f), totalDistance(0), totalDuration(0), serviceTripDuration(0), serviceTripCosts(0), time()
{
}


CirculationStats::~CirculationStats()
{
}
