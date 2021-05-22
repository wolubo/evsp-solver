#pragma once

#include <memory>
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuEmptyTrips.h"
#include "CuStops.h"
#include "CuMatrix1.hpp"
#include "EvspLimits.h"



/// <summary>
/// Matrix, aus der sich die ID der Verbindungsfahrt zwischen zwei Haltestellen entnehmen lässt.
/// Die Zellen enthalten eine ungültige EmptyTripId, falls keine Verbindungsfahrt existiert, die beide Haltestellen verbindet 
/// und andernfalls die Id der entsprechenden Verbindungsfahrt.
/// </summary>
class ConnectionMatrix
{
public:
	ConnectionMatrix() = delete;
	ConnectionMatrix(CuEmptyTrips *emptyTrips, int numOfStops);
	~ConnectionMatrix();

	ConnectionMatrix* getDevPtr();
	void copyToHost();

	CU_HSTDEV EmptyTripId getEmptyTripId(StopId from, StopId to) const;
	CU_HSTDEV bool connected(StopId from, StopId to) const;
	shared_ptr<CuEmptyTrip> getEmptyTrip(StopId from, StopId to, bool createFake = true) const;
	CU_HSTDEV bool getEmptyTrip(StopId from, StopId to, CuEmptyTrip& emptyTrip, bool createFake = true) const;

private:
	void setAll(EmptyTripId value);

	CU_HSTDEV int getNumOfRows() { return _matrix->getNumOfRows(); }
	CU_HSTDEV int getNumOfCols() { return _matrix->getNumOfCols(); }

	CU_HSTDEV StopId lastFromStopId() const;
	CU_HSTDEV StopId lastToStopId() const;

	CU_HSTDEV void set(StopId from, StopId to, EmptyTripId value);
	CU_HSTDEV EmptyTripId get(StopId from, StopId to) const;
	CU_HSTDEV EmptyTripId& itemAt(StopId from, StopId to);

	CuMatrix1<EmptyTripId> *_matrix;
	CuEmptyTrips *_emptyTrips;
	ConnectionMatrix *_devicePtr;
};

