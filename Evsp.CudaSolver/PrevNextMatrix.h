#pragma once

#include <memory>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuMatrix1.hpp"
#include "CuServiceTrips.h"
#include "CuEmptyTrips.h"
#include "ConnectionMatrix.h"
#include "PrevNextMatrix.h"

using namespace std;

class CuProblem;

/// <summary>
/// Erzeugt eine Matrix, aus der sich die L�nge des Intervalls ablesen l�sst, das zwischen den beiden
/// Servicefahrten A und B liegt. Ein solches Intervall existiert nur dann, wenn die Servicefahrt A endet, 
/// bevor die Servicefahrt B beginnt. Ausserdem wird unterstellt, dass B tats�chlich nach A durchgef�hrt 
/// werden kann (dass also die Endhaltestelle von A gleich der Starthaltestelle von B ist oder das eine
/// Verbindungsfahrt existiert, die innerhalb des Intervalls zwischen A und B durchgef�hrt werden kann).
///
/// Neben der L�nge des Intervalls zwischen A und B ist auch noch die um die Dauer der ggf. n�tigen
/// Verbindungsfahrt verminderte Intervalll�nge gespeichert.
///
/// Bedeutung der gespeicherten Werte:
///   -1	A endet erst, nachdem B begonnen hat (A und B �berschneiden sich zeitlich oder B findet 
///         bereits vor A statt).
///  >=0	L�nge des Intervalls zwischen A und B in Sekunden
///
/// Bedeutung der Indizes:
/// Zeilen (row):  ID der Servicefahrt A (also der zuerst Fahrten, die zuerst ausgef�hrt werden sollen).
/// Spalten (col): ID der Servicefahrt B.
/// </summary>
class PrevNextMatrix
{
public:
	PrevNextMatrix(CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix, PlattformConfig plattform, bool performChecks);
	~PrevNextMatrix();

	PrevNextMatrix* getDevPtr();
	void copyToHost();

	/// <summary>
	/// Pr�ft, ob tripB nach tripA durchgef�hrt werden kann. Dabei werden sowohl die Abfahrts- und Ankunfstszeiten als auch die
	/// Dauer einer ggf. n�tigen Verbindungsfahrt ber�cksichtigt
	/// </summary>
	CU_HSTDEV bool checkCombination(ServiceTripId tripA, ServiceTripId tripB) const;

	/// <summary>
	/// Liefert die Dauer des Intervalls, dass zwischen der Ankunft der Fahrt A und der 
	/// Abfahrt der Fahrt B liegt.
	/// </summary>
	CU_HSTDEV DurationInSeconds getInterval(ServiceTripId tripA, ServiceTripId tripB) const;

	/// <summary>
	/// Liefert eine Liste der Servicefahrten, die auf die �bergebene Fahrt folgen k�nnen. Also eine Liste m�glicher Nachfolger der Servicefahrt.
	/// </summary>
	shared_ptr<CuVector1<ServiceTripId>> collectSuccessors(ServiceTripId trip) const;

	/// <summary>
	/// Liefert eine Liste der Servicefahrten, die auf die die �bergebene Fahrt folgen k�nnen. Also eine Liste m�glicher Vorg�nger der Servicefahrt.
	/// </summary>
	shared_ptr<CuVector1<ServiceTripId>> collectPredecessors(ServiceTripId trip) const;

private:
	void createCpu(CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix, int numOfServiceTrips);
	void createGpu(CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix, int numOfServiceTrips);

	/// <summary>
	/// �berpr�ft, ob die Matrix inhaltlich korrekt ist.
	/// </summary>
	/// <returns>True, wenn in der Matrix keine Fehler gefunden wurden. Andernfalls false.</returns>
	bool check(CuServiceTrips *serviceTrips, CuEmptyTrips *emptyTrips, ConnectionMatrix *connectionMatrix);

	CuMatrix1<DurationInSeconds> *_pnMatrix;

	PrevNextMatrix *_devicePtr;
};

