#include "RandomCpu.h"



RandomCpu::RandomCpu()
{
	random_device rd;
	_intGenerator = mt19937(rd());
	_doubleGenerator = uniform_real_distribution<double>(numeric_limits<double>::epsilon(), 1);
}


RandomCpu::~RandomCpu()
{
}


int RandomCpu::rand(int maxValue)
{
	return _intGenerator() % maxValue;
}


bool RandomCpu::shot(float hitChance)
{
	assert(hitChance >= 0.0f);
	assert(hitChance <= 1.0f);
	if (hitChance <= 0.0f) return false;
	if (hitChance >= 1.0f) return true;
	return _doubleGenerator(_intGenerator) <= hitChance;
}


unsigned int RandomCpu::weightedRandomSelection(unsigned int numOfChoices, const CuVector1<float> &weights, float totalWeight)
{
	assert(numOfChoices > 0);

	if (numOfChoices == 1) return 0;

	float r = (float)(_doubleGenerator(_intGenerator) * totalWeight);

	float f = 0.0f;
	for (unsigned int i = 0; i < numOfChoices; i++) {
		f += weights.get(i);
		if (f >= r) return i;
	}

	return -1;
}


unsigned int RandomCpu::weightedRandomSelection(unsigned int numOfChoices, const CuVector1<float> &weights)
{
	float totalWeight = 0.0f;
	for (unsigned int i = 0; i < numOfChoices; i++) {
		assert(weights.get(i) > 0.0f);
		totalWeight += weights.get(i);
	}
	unsigned int retVal = weightedRandomSelection(numOfChoices, weights, totalWeight);
	assert(retVal < numOfChoices);
	return retVal;
}


shared_ptr<CuVector1<int>> RandomCpu::shuffle(int size)
{
	shared_ptr<CuVector1<int>> retVal = make_shared<CuVector1<int>>(size);
	for (int i=0; i < size; i++) retVal->set(i, i);

	// Fisher-Yates-Shuffle
	for (int i = size - 1; i > 0; i--) {
		int index = rand(i);
		int a = retVal->get(index);
		int b = retVal->get(i);
		retVal->set(i, a);
		retVal->set(index, b);
	}

	return retVal;
}
