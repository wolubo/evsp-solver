#pragma once

//#ifdef CudaModel_EXPORTS
//#define BinaryMatrix_API __declspec(dllexport)
//#else
//#define BinaryMatrix_API __declspec(dllimport)
//#endif

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"



typedef int t_CellType;

class BinaryMatrix;

class /*BinaryMatrix_API*/ BinaryMatrixDevice
{
public:
	static BinaryMatrixDevice* create(int numOfRows, int numOfCols);
	static BinaryMatrixDevice* createCheckerboardPattern(int numOfRows, int numOfCols);
	static void destroy(BinaryMatrixDevice* theMatrix);
	BinaryMatrixDevice(BinaryMatrixDevice&) = delete;
	BinaryMatrixDevice& operator=(const BinaryMatrixDevice &rhs) { assert(false); return *this; }
	CU_HSTDEV int getNumOfRows() { return _numOfRows; }
	CU_HSTDEV int getNumOfCols() { return _numOfCols; }
	bool get(int row, int col);
	CU_DEV void set(int row, int col);
	CU_DEV void unset(int row, int col);
	CU_DEV void toggle(int row, int col);
	BinaryMatrixDevice* getDevPtr();
	BinaryMatrix* copyToHost();
protected:
	void init(int numOfRows, int numOfCols);
	CU_HSTDEV void calculateIndex(int row, int col, int &arrayIndex, int &bitIndex);
	BinaryMatrixDevice() {}
	~BinaryMatrixDevice() {}
	int _bitsPerCell;	// Anzahl der Bits, die sich in einer Array-Zelle unterbringen lassen.
	int _numOfRows;		// Anzahl der Bits in einer Spalte.
	int _numOfCols;		// Anzahl der Bits in einer Zeile.
	int _arraySize;		// Anzahl der Array-Zellen.
	t_CellType* _data;	// Anfang des Arrays.
};

class /*BinaryMatrix_API*/ BinaryMatrix : public BinaryMatrixDevice
{
public:
	static BinaryMatrix* create(int numOfRows, int numOfCols);
	static BinaryMatrix* create(int arraySize, int bitsPerCell, t_CellType* data, int numOfCols, int numOfRows);
	static void destroy(BinaryMatrix* hostPtr);
	BinaryMatrix(BinaryMatrix&) = delete;
	BinaryMatrix& operator=(const BinaryMatrix &rhs) { assert(false); return *this; }
	void setAll();
	void unsetAll();
	void set(int row, int col);
	void unset(int row, int col);
	void toggle(int row, int col);
	bool operator==(BinaryMatrix &other);
	bool operator!=(BinaryMatrix &other);
private:
	BinaryMatrix() {}
	~BinaryMatrix() {}
};


