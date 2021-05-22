//#define TEST_DoubleLinkedList

#ifdef TEST_DoubleLinkedList

/*
Diese Datei enthält Code zum Testen der Klassen der CudaSolver-Lib.
Zum Ausführen des Test-Codes muss die CudaSolver als Exe übersetzt (Eigenschaften -> Konfigurationstyp) und als Startprojekt festgelegt werden.
*/

#include <conio.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DoubleLinkedList2.hpp"


void handleError(cudaError_t code)
{
	if (code != cudaSuccess) {
		printf("ERROR: %s (Code: %d) %s %d\n", cudaGetErrorString(code), code, __FILE__, __LINE__);
		exit(code);
	}
}

__global__ void checkDoubleLinkedList() 
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	DoubleLinkedList2<int> *dll = new DoubleLinkedList2<int>(10, 10);
	dll->append(1);
	for (int i = 0; i < 100; i++) {
		dll->append(i);
		dll->insertItemBeforeCurrent(i);
		dll->gotoFirst();
		dll->insertItemAfterCurrent(i);
		dll->gotoNext();
	}
	delete dll;
}

int main()
{
	handleError(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000000000));

	const int numOfThreads = 2048;

	handleError(cudaSetDevice(0));

	const int blockSize = 512;
	int numOfBlocks = (numOfThreads + blockSize - 1) / blockSize;
	dim3 dimGrid(numOfBlocks);
	dim3 dimBlock(blockSize);
	checkDoubleLinkedList << <dimGrid, dimBlock >> > ();

	handleError(cudaDeviceReset());
}
#endif