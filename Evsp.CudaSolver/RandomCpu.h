#pragma once

#include <random>
#include <memory>
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuVector1.hpp"

using namespace std;

class RandomCpu
{
public:
	RandomCpu();
	~RandomCpu();

public:
	int rand(int maxValue);

	/// <summary>
	/// Simuliert einen Zufallsauswahl mit einer definierten Trefferchance. Liefert true, wenn die Wahl erfolgreich war.
	/// Das ist dann der Fall, wenn die ermittelte Zufallszahl r (0.0 < r <= 1.0) unter 'hitChance' liegt. 
	/// </summary>
	bool shot(float hitChance);

	/// <summary>
	/// Trifft eine gewichtete Zufallsauswahl.
	/// </summary>
	unsigned int weightedRandomSelection(unsigned int numOfChoices, const CuVector1<float> &weights);
	unsigned int weightedRandomSelection(unsigned int numOfChoices, const CuVector1<float> &weights, float totalWeight);

	shared_ptr<CuVector1<int>> shuffle(int size);

private:
	mt19937 _intGenerator;
	uniform_real_distribution<double> _doubleGenerator;
};

