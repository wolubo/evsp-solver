//#define TEST_RandomCuda

#ifdef TEST_RandomCuda

/*
Diese Datei enthält Code zum Testen der Klassen der CudaSolver-Lib.
Zum Ausführen des Test-Codes muss die CudaSolver als Exe übersetzt (Eigenschaften -> Konfigurationstyp) und als Startprojekt festgelegt werden.
*/

#include <conio.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "RandomGpu.h"

using namespace WoBo::EVSP::CudaSolver;


__global__ void checkCudaRandom(RandomGpu *rnd_dev, int numOfThreads)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < numOfThreads) {
		int rnd_number = rnd_dev->rand(500, id);
	}
}


int main()
{
	const int numOfThreads = 1000000;
	const int numOfRounds = 2;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t code = cudaSetDevice(0);
	if (code != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(code);
	}

	for (int i = 0; i < numOfRounds; i++) {
	
		RandomGpu *rnd_dev = RandomGpu::create(numOfThreads);

	//const int blockSize = 1024;
	//int numOfBlocks = (numOfThreads + blockSize - 1) / blockSize;
	//dim3 dimGrid(numOfBlocks);
	//dim3 dimBlock(blockSize);
	//checkCudaRandom << <dimGrid, dimBlock >> > (rnd_dev, numOfThreads);
		//CUDA_CHECK(cudaGetLastError());
		//cudaError_t code = cudaDeviceSynchronize();
	//if (code != cudaSuccess) {
	//	printf("ERROR: %s\n", cudaGetErrorString(code));
	//	exit(code);
	//}

		RandomGpu::destroy(rnd_dev);
	}
	
	code = cudaDeviceReset();
	if (code != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

}

#endif