#pragma once

#include "curand.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuVector1.hpp"

//typedef curandState EvspCurandState;
typedef curandStatePhilox4_32_10_t EvspCurandState;
//typedef curandStateMRG32k3a EvspCurandState;

class RandomGpu
{
public:
	RandomGpu(int size);
	~RandomGpu();

	RandomGpu* getDevPtr();

	CU_DEV int rand(int maxValue, int threadId);

	/// <summary>
	/// Simuliert einen Zufallsauswahl mit einer definierten Trefferchance. Liefert true, wenn die Wahl erfolgreich war.
	/// Das ist dann der Fall, wenn die ermittelte Zufallszahl r (0.0 < r <= 1.0) unter 'hitChance' liegt. 
	/// </summary>
	CU_DEV bool shot(float hitChance, int threadId);

	/// <summary>
	/// Trifft eine gewichtete Zufallsauswahl.
	/// </summary>
	CU_DEV unsigned int weightedRandomSelection(unsigned int numOfChoices, const CuVector1<float> &weights, int threadId);
	CU_DEV unsigned int weightedRandomSelection(unsigned int numOfChoices, const CuVector1<float> &weights, float totalWeight, int threadId);

	CU_HSTDEV int getSize() { return _size; }

private:
	EvspCurandState *_states;
	int _size;
	RandomGpu* _devPtr;
};
