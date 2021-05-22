//#define TEST_RandomMatrix
#ifdef TEST_RandomMatrix

/*
Diese Datei enthält Code zum Testen der Klassen der CudaSolver-Lib.
Zum Ausführen des Test-Codes muss die CudaSolver als Exe übersetzt (Eigenschaften -> Konfigurationstyp) und als Startprojekt festgelegt werden.
*/

#include <iostream>
#include <stdlib.h>
#include <conio.h>
#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "RandomGpu.h"
#include "RandomMatrix.h"

#include "CuMatrix2.hpp"

using namespace WoBo::EVSP::CudaSolver;




void handleError(cudaError_t code)
{
	if (code != cudaSuccess) {
		printf("ERROR: %s (Code: %d) %s %d\n", cudaGetErrorString(code), code, __FILE__, __LINE__);
		exit(code);
	}
}

//__global__ void initKernelWhitoutMatrix(int *retVal, int numOfRows, int numOfCols, int initValue)
//{
//	int row = blockIdx.x * blockDim.x + threadIdx.x;
//	int col = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (row == 0 && col == 0) {
//		printf("ALPHA\n");
//	}
//
//	if (row == numOfRows - 1 && col == numOfCols - 1) {
//		printf("OMEGA\n");
//	}
//
//	if (row < numOfRows && col < numOfCols) {
//		retVal[row * numOfCols + col] = initValue;
//	}
//}


int main()
{
	const int numOfRows = 1000;
	const int numOfCols = 1000;

	// Choose which GPU to run on, change this on a multi-GPU system.
	handleError(cudaSetDevice(0));

	RandomMatrix *rndMat = RandomMatrix::create(numOfRows, numOfCols);
	std::cout << RandomMatrix::check(rndMat, numOfRows, numOfCols) << std::endl;

	RandomMatrix::destroy(rndMat);

	handleError(cudaDeviceReset());
}

#endif