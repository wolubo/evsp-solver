#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"

///<summary>
/// Realisiert einen Lock-Mechanismus für Matrizen variabler Größe. 
///</summary>
class CuLockMatrix1
{
public:
	CuLockMatrix1(int numOfRows, int numOfCols);
	~CuLockMatrix1();
	CuLockMatrix1* getDevPtr();
	CU_DEV bool lock(int row, int col);
	CU_DEV void unlock(int row, int col);
private:
	int _numOfRows;
	int _numOfCols;
	int *_lock;
	CuLockMatrix1 *_devPtr;
};
