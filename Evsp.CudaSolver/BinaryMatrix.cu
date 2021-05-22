#include <mutex>
//#include <limits>
#include <assert.h>
#include <stdexcept>
#include <mutex>
#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaCheck.h"
#include "BinaryMatrix.h"




std::mutex bm_mutex;

__global__ void checkerboardPatternKernel(BinaryMatrixDevice* bcm);


void BinaryMatrix::setAll()
{
	for (int i = 0; i < _arraySize; i++) _data[i] = ~((t_CellType)0);
}

void BinaryMatrix::unsetAll()
{
	for (int i = 0; i < _arraySize; i++) _data[i] = 0;
}

CU_DEV void BinaryMatrixDevice::set(int row, int col)
{
	if (_data == 0 || row < 0 || row >= _numOfRows || col < 0 || col >= _numOfCols) return;

	int arrayIndex, bitIndex;

	calculateIndex(row, col, arrayIndex, bitIndex);

	atomicOr(&_data[arrayIndex], (t_CellType)1 << bitIndex);
}

void BinaryMatrix::set(int row, int col)
{
	if (row < 0 || row >= _numOfRows) throw  std::invalid_argument("Error in BinaryMatrix::Set: row out of bounds!");
	if (col < 0 || col >= _numOfCols) throw  std::invalid_argument("Error in BinaryMatrix::Set: col out of bounds!");

	int arrayIndex, bitIndex;

	calculateIndex(row, col, arrayIndex, bitIndex);

	bm_mutex.lock();
	_data[arrayIndex] = _data[arrayIndex] | ((t_CellType)1 << bitIndex);
	bm_mutex.unlock();
}

CU_DEV void BinaryMatrixDevice::unset(int row, int col)
{
	if (_data == 0 || row < 0 || row >= _numOfRows || col < 0 || col >= _numOfCols) return;

	int arrayIndex, bitIndex;

	calculateIndex(row, col, arrayIndex, bitIndex);

	atomicAnd(&_data[arrayIndex], ~((t_CellType)1 << bitIndex));
}

void BinaryMatrix::unset(int row, int col)
{
	if (row < 0 || row >= _numOfRows) throw  std::invalid_argument("Error in BinaryMatrix::Unset: row out of bounds!");
	if (col < 0 || col >= _numOfCols) throw  std::invalid_argument("Error in BinaryMatrix::Unset: col out of bounds!");

	int arrayIndex, bitIndex;

	calculateIndex(row, col, arrayIndex, bitIndex);

	bm_mutex.lock();
	_data[arrayIndex] = _data[arrayIndex] & ~((t_CellType)1 << bitIndex);
	bm_mutex.unlock();
}

CU_DEV void BinaryMatrixDevice::toggle(int row, int col)
{
	if (_data == 0 || row < 0 || row >= _numOfRows || col < 0 || col >= _numOfCols) return;
	int arrayIndex, bitIndex;
	calculateIndex(row, col, arrayIndex, bitIndex);
	atomicXor(&_data[arrayIndex], (t_CellType)1 << bitIndex);
}

void BinaryMatrix::toggle(int row, int col)
{
	if (row < 0 || row >= _numOfRows) throw  std::invalid_argument("Error in BinaryMatrix::Toggle: row out of bounds!");
	if (col < 0 || col >= _numOfCols) throw  std::invalid_argument("Error in BinaryMatrix::Toggle: col out of bounds!");

	int arrayIndex, bitIndex;

	calculateIndex(row, col, arrayIndex, bitIndex);

	bm_mutex.lock();
	_data[arrayIndex] = _data[arrayIndex] ^ ((t_CellType)1 << bitIndex);
	bm_mutex.unlock();
}

bool BinaryMatrix::operator==(BinaryMatrix &other)
{
	if (_numOfRows != other._numOfRows) return false;
	if (_numOfCols != other._numOfCols) return false;
	for (int i = 0; i < _arraySize; i++) {
		if (_data[i] != other._data[i]) return false;
	}
	return true;
}

bool BinaryMatrix::operator!=(BinaryMatrix &other)
{
	return !(this->operator==(other));
}


void BinaryMatrixDevice::init(int numOfRows, int numOfCols)
{
	_numOfRows = numOfRows; // Anzahl der Bits in einer Spalte.
	if (_numOfRows < 0) _numOfRows = 0;

	_numOfCols = numOfCols; // Anzahl der Bits in einer Zeile.
	if (_numOfCols < 0) _numOfCols = 0;

	_bitsPerCell = 8 * sizeof(t_CellType); // Bits pro Array-Element.

	int totalNumberOfBits = numOfRows*numOfCols; // Gesamtanzahl der Bits in der CuMatrix2.
	_arraySize = totalNumberOfBits / _bitsPerCell;
	if (totalNumberOfBits % _bitsPerCell != 0) {
		// Falls beim Berrechnen der Arraygröße ein Rest bleibt muss ein zusätzliches Element reserviert werden, um auch die restlichen Bits aufzunehmen.
		_arraySize++;
	}

	_data = 0;
}

bool BinaryMatrixDevice::get(int row, int col)
{
	if (_data == 0 || row < 0 || row >= _numOfRows || col < 0 || col >= _numOfCols) return false;
	int arrayIndex, bitIndex;
	calculateIndex(row, col, arrayIndex, bitIndex);
	t_CellType mask = (t_CellType)1 << bitIndex;
	t_CellType value = _data[arrayIndex] & mask;
	return (value > 0);
}

CU_HSTDEV inline void BinaryMatrixDevice::calculateIndex(int row, int col, int &arrayIndex, int &bitIndex)
{
	int bitNumber = row*_numOfCols + col;
	arrayIndex = bitNumber / _bitsPerCell;
	bitIndex = bitNumber % _bitsPerCell;
}

BinaryMatrix* BinaryMatrix::create(int numOfRows, int numOfCols)
{
	if (numOfRows <= 0) throw  std::invalid_argument("numOfRows <= 0");
	if (numOfCols <= 0) throw  std::invalid_argument("numOfCols <= 0");

	BinaryMatrix* retVal;
	CUDA_CHECK(cudaMallocHost((void**)&(retVal), sizeof(BinaryMatrix)));

	retVal->init(numOfRows, numOfCols);

	CUDA_CHECK(cudaMallocHost((void**)&(retVal->_data), sizeof(t_CellType) * retVal->_arraySize));

	retVal->unsetAll();

	return retVal;
}

BinaryMatrix* BinaryMatrix::create(int arraySize, int bitsPerCell, t_CellType* data, int numOfCols, int numOfRows)
{
	BinaryMatrix* retVal;
	CUDA_CHECK(cudaMallocHost((void**)&(retVal), sizeof(BinaryMatrix)));
	retVal->_arraySize = arraySize;
	retVal->_bitsPerCell = bitsPerCell;
	retVal->_data = data;
	retVal->_numOfCols = numOfCols;
	retVal->_numOfRows = numOfRows;
	return retVal;
}

void BinaryMatrix::destroy(BinaryMatrix* hostPtr)
{
	if (hostPtr != 0) {
		CUDA_CHECK(cudaFreeHost(hostPtr));
	}
}

BinaryMatrixDevice* BinaryMatrixDevice::create(int numOfRows, int numOfCols)
{
	BinaryMatrixDevice* retVal;

	// Stellvertreter-Objekt auf Host erzeugen, dessen Inhalt in den Device-Speicher kopiert werden kann.
	BinaryMatrixDevice temp;
	temp.init(numOfRows, numOfCols);

	// Speicher für Device-Objekt reservieren.
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(retVal), sizeof(BinaryMatrixDevice)));
	

	// Device-Speicher für die Daten reservieren und Pointer darauf ins Stellvertreter-Objekt eintragen.
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(temp._data), sizeof(t_CellType) * temp._arraySize));
	

	// Daten aus Stellvertreter-Objekt ins Device-Objekt kopieren.
	CUDA_CHECK(cudaMemcpy(retVal, &temp, sizeof(BinaryMatrixDevice), cudaMemcpyHostToDevice));

	return retVal;
}

__global__ void checkerboardPatternKernel(BinaryMatrixDevice* bcm)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int numOfRows = bcm->getNumOfRows();
	int bitNumber = row*numOfRows + col;
	if (bitNumber % 2 == 0) {
		bcm->set(row, col);
	}
	else {
		bcm->unset(row, col);
	}
}

BinaryMatrixDevice* BinaryMatrixDevice::createCheckerboardPattern(int numOfRows, int numOfCols)
{
	const int block_size = 32;

	BinaryMatrixDevice* retVal = BinaryMatrixDevice::create(numOfRows, numOfCols);

	CUDA_CHECK(cudaSetDevice(0));

	int numOf_x_Blocks = numOfRows / block_size;
	int numOf_y_Blocks = numOfCols / block_size;

	if (numOfRows % block_size != 0) numOf_x_Blocks++;
	if (numOfCols % block_size != 0) numOf_y_Blocks++;

	dim3 dimGrid(numOf_x_Blocks, numOf_y_Blocks);
	dim3 dimBlock(block_size, block_size);

	checkerboardPatternKernel << <dimGrid, dimBlock >> > (retVal);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	return retVal;
}

BinaryMatrixDevice* BinaryMatrixDevice::getDevPtr()
{
	BinaryMatrixDevice* retVal;
	BinaryMatrixDevice temp;
	temp.init(_numOfRows, _numOfCols);
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(retVal), sizeof(BinaryMatrixDevice)));
	
	CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&(temp._data), sizeof(t_CellType) * temp._arraySize));
	
	CUDA_CHECK(cudaMemcpy(retVal, &temp, sizeof(BinaryMatrixDevice), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(retVal, _data, sizeof(t_CellType) * _arraySize, cudaMemcpyHostToDevice));
	return retVal;
}

BinaryMatrix* BinaryMatrixDevice::copyToHost()
{
	// ACHTUNG: Die Member des Objekts befinden sich im Device-Speicher!

	// Device-Objekt in den Host-Speicher holen.
	BinaryMatrixDevice *deviceObject;
	CUDA_CHECK(cudaMallocHost((void**)&(deviceObject), sizeof(BinaryMatrixDevice)));
	CUDA_CHECK(cudaMemcpy(deviceObject, this, sizeof(BinaryMatrixDevice), cudaMemcpyDeviceToHost));

	// Daten des Device-Objekts in den Host-Speicher holen.
	t_CellType *data;
	CUDA_CHECK(cudaMallocHost((void**)&(data), sizeof(t_CellType) * deviceObject->_arraySize));
	CUDA_CHECK(cudaMemcpy(data, deviceObject->_data, sizeof(t_CellType) * deviceObject->_arraySize, cudaMemcpyDeviceToHost));

	// Host-Objekt erzeugen und initialisieren.
	BinaryMatrix* retVal = BinaryMatrix::create(deviceObject->_arraySize, deviceObject->_bitsPerCell, data, deviceObject->_numOfCols, deviceObject->_numOfRows);

	// Device-Objekt im Host-Speicher freigeben.
	CUDA_CHECK(cudaFreeHost(deviceObject));

	return retVal;
}

void BinaryMatrixDevice::destroy(BinaryMatrixDevice* devicePtr)
{
	if (devicePtr != 0) {
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, devicePtr));
	}
}
