#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"


///<summary>
/// Realisiert einen Lock-Mechanismus f�r Vektoren variabler Gr��e. 
///</summary>
class CuLockVector1
{
public:
	CuLockVector1(int size);
	~CuLockVector1();
	CuLockVector1* getDevPtr();
	CU_DEV bool lock(int index);
	CU_DEV void unlock(int index);
private:
	int _size;
	int *_lock;
	CuLockVector1 *_devPtr;
};

