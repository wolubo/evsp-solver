#include "RandomGpu.h"

#include <stdio.h>
#include <assert.h>
#include <time.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "curand_kernel.h"

#include "EVSP.BaseClasses/Typedefs.h"
#include "CudaCheck.h"



__global__ void curandInitKernel(unsigned long long seed, EvspCurandState* states, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		curand_init(seed, id, 0, &states[id]);
	}
}


RandomGpu::RandomGpu(int size)
 : _size(size), _devPtr(0)
{
	assert(size >= 0);

	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_states, size * sizeof(EvspCurandState)));
	
	int blockSize;
	int minGridSize;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)curandInitKernel));
	int numOfBlocks = (size + blockSize - 1) / blockSize;
	dim3 dimGrid(numOfBlocks);
	dim3 dimBlock(blockSize);
	curandInitKernel << <dimGrid, dimBlock >> > (time(0), _states, _size);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&_devPtr, sizeof(RandomGpu)));
	
	CUDA_CHECK(cudaMemcpy(_devPtr, this, sizeof(RandomGpu), cudaMemcpyHostToDevice));
}


RandomGpu::~RandomGpu()
{
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devPtr));
	CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _states));
}


RandomGpu* RandomGpu::getDevPtr()
{
	return _devPtr;
}


__device__ int RandomGpu::rand(int maxValue, int threadId)
{
	if (threadId >= _size) return 0;
	return curand(&_states[threadId]) % maxValue;
}


__device__ bool RandomGpu::shot(float hitChance, int threadId)
{
	assert(hitChance >= 0.0f);
	assert(hitChance <= 1.0f);
	if (hitChance <= 0.0f) return false;
	if (hitChance >= 1.0f) return true;
	return curand_uniform(&_states[threadId]) <= hitChance;
}


CU_DEV unsigned int RandomGpu::weightedRandomSelection(unsigned int numOfChoices, const CuVector1<float> &weights, float totalWeight, int threadId)
{
	assert(numOfChoices > 0);

	if (numOfChoices == 1) return 0;

	float r = curand_uniform(&_states[threadId]) * totalWeight;

	float f = 0.0f;
	for (unsigned int i = 0; i < numOfChoices; i++) {
		f += weights.get(i);
		if (f >= r) return i;
	}

	assert(false);
	return 0;
}


CU_DEV unsigned int RandomGpu::weightedRandomSelection(unsigned int numOfChoices, const CuVector1<float> &weights, int threadId)
{
	float totalWeight = 0.0f;
	for (unsigned int i = 0; i < numOfChoices; i++) {
		assert(weights.get(i) > 0.0f);
		totalWeight += weights.get(i);
	}
	unsigned int retVal = weightedRandomSelection(numOfChoices, weights, threadId);
	assert(retVal < numOfChoices);
	return retVal;
}
