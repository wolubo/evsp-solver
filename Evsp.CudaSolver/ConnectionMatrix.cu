#include "ConnectionMatrix.h"
#include "EVSP.BaseClasses/Stopwatch.h"



ConnectionMatrix::ConnectionMatrix(CuEmptyTrips *emptyTrips, int numOfStops)
	: _devicePtr(0), _matrix(0), _emptyTrips(emptyTrips)
{
	Stopwatch stopwatch;
	stopwatch.start();

	_matrix = new CuMatrix1<EmptyTripId>(numOfStops, numOfStops);
	_matrix->setAll(EmptyTripId::invalid());
	for (EmptyTripId i(0); i <= emptyTrips->lastId(); i++) {
		CuEmptyTrip et = emptyTrips->getEmptyTrip(i);
		_matrix->set((short)et.fromStopId, (short)et.toStopId, i);
	}

	stopwatch.stop("ConnectionMatrix erzeugt (CPU): ");
}


ConnectionMatrix::~ConnectionMatrix()
{
	if (_matrix) delete _matrix;
	if (_devicePtr)
	{
		CUDA_CHECK(wbCudaFree(__FILE__, __LINE__, _devicePtr));
	}
}

ConnectionMatrix* ConnectionMatrix::getDevPtr()
{
	if (!_devicePtr) {
		CuMatrix1<EmptyTripId> *tempMatrix = _matrix;
		CuEmptyTrips *tempEmptyTrips = _emptyTrips;

		_matrix = _matrix->getDevPtr();
		_emptyTrips = _emptyTrips->getDevPtr();
		ConnectionMatrix* newDevicePtr = 0;

		CUDA_CHECK(wbCudaMalloc(__FILE__, __LINE__, (void**)&newDevicePtr, sizeof(ConnectionMatrix)));
		CUDA_CHECK(cudaMemcpy(newDevicePtr, this, sizeof(ConnectionMatrix), cudaMemcpyHostToDevice));

		_matrix = tempMatrix;
		_emptyTrips = tempEmptyTrips;
		_devicePtr = newDevicePtr;
	}
	return _devicePtr;
}


void ConnectionMatrix::copyToHost()
{
	if (_devicePtr) {
		_matrix->copyToHost();
	}
}


CU_HSTDEV void ConnectionMatrix::set(StopId from, StopId to, EmptyTripId value)
{
	_matrix->set((short)from, (short)to, value);
}


CU_HSTDEV EmptyTripId ConnectionMatrix::get(StopId from, StopId to) const
{
	return _matrix->get((short)from, (short)to);
}


CU_HSTDEV EmptyTripId& ConnectionMatrix::itemAt(StopId from, StopId to)
{
	return _matrix->itemAt((short)from, (short)to);
}


void ConnectionMatrix::setAll(EmptyTripId value)
{
	_matrix->setAll(value);
}


CU_HSTDEV EmptyTripId ConnectionMatrix::getEmptyTripId(StopId from, StopId to) const
{
	assert(from.isValid());
	assert(to.isValid());
	return _matrix->get((short)from, (short)to);
}


CU_HSTDEV bool ConnectionMatrix::connected(StopId from, StopId to) const
{
	assert(from.isValid());
	assert(to.isValid());
	if (from != to) {
		return getEmptyTripId(from, to).isValid();
	}
	return true;
}


shared_ptr<CuEmptyTrip> ConnectionMatrix::getEmptyTrip(StopId from, StopId to, bool createFake) const
{
	shared_ptr<CuEmptyTrip> retVal;
	if (from == to && createFake) {
		retVal = make_shared<CuEmptyTrip>(from, to, DistanceInMeters(0), DurationInSeconds(0));
	}
	else {
		EmptyTripId id = getEmptyTripId(from, to);
		if (id.isValid()) {
			retVal = make_shared<CuEmptyTrip>(_emptyTrips->getEmptyTrip(id));
		}
	}
	return retVal;
}


bool ConnectionMatrix::getEmptyTrip(StopId from, StopId to, CuEmptyTrip& emptyTrip, bool createFake) const
{
	bool retVal = false;
	if (from == to && createFake) {
		emptyTrip = CuEmptyTrip(from, to, DistanceInMeters(0), DurationInSeconds(0));
		retVal = true;
	}
	else {
		EmptyTripId id = getEmptyTripId(from, to);
		if (id.isValid()) {
			emptyTrip = CuEmptyTrip(_emptyTrips->getEmptyTrip(id));
			retVal = true;
		}
	}
	return retVal;
}



CU_HSTDEV StopId ConnectionMatrix::lastFromStopId() const
{
	int n = _matrix->getNumOfRows();
	if (n > 0)
		return StopId(n - 1);
	else
		return StopId::invalid();
}

CU_HSTDEV StopId ConnectionMatrix::lastToStopId() const
{
	int n = _matrix->getNumOfCols();
	if (n > 0)
		return StopId(n - 1);
	else
		return StopId::invalid();
}
