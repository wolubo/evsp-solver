#pragma once

#ifdef __CUDACC__
#include "device_launch_parameters.h"
#define CU_HSTDEV __host__ __device__
#define CU_DEV __device__
#define CU_GLOBAL __global__
#else
#define CU_HSTDEV
#define CU_DEV
#define CU_GLOBAL
#endif