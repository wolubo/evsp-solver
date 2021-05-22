#pragma once

#include "EVSP.BaseClasses\DeviceLaunchParameters.h"

//#define WB_MEMCHECK

#define ERRORMSG(msg) printf("Error: %s (%s %d)\n", (msg), __FILE__, __LINE__);

#ifdef __CUDACC__

/// <summary>
/// Wrapper für cudaMalloc()
/// Wenn WB_MEMCHECK definiert ist wird die Adresse des gerade zugewiesenen Speicherbereichs auf der Konsole ausgegeben, um das auffinden
/// von Memoryleaks zu ermöglichen.
/// </summary>
cudaError_t wbCudaMalloc(const char *file, int line, void **devPtr, size_t size);

/// <summary>
/// Wrapper für cudaFree()
/// Wenn WB_MEMCHECK definiert ist wird die Adresse des gerade zugewiesenen Speicherbereichs auf der Konsole ausgegeben, um das auffinden
/// von Memoryleaks zu ermöglichen.
/// </summary>
cudaError_t wbCudaFree(const char *file, int line, void *devPtr);

#ifdef _DEBUG
#define MEMSTAT printMemStat(__FILE__, __LINE__);
#else
#define MEMSTAT
#endif

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK_DEV(ans) { gpuAssertDev((ans), __FILE__, __LINE__); }

void printMemStat(const char *file, int line);
__host__ void gpuAssert(cudaError_t code, const char *file, int line);
__device__ void gpuAssertDev(cudaError_t code, const char *file, int line);

#else

#define MEMSTAT
#define CUDA_CHECK
#define CUDA_CHECK_DEV

#endif
