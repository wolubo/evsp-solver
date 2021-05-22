#pragma once

#include <memory>
#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/ConfigSettings.h"
#include "CuServiceTrips.h"
#include "CuEmptyTrips.h"
#include "CuStops.h"
#include "CuMatrix1.hpp"

using namespace std;

class CuProblem;

/// <summary>
/// Matrix, aus der sich die ID der Ladestation ablesen lässt, die sich auf dem Weg zwischen zwei 
/// Haltestellen in der kürzesten Zeit erreichen lässt.
/// Die Zeilen der Matrix ('row') entsprechen den IDs der Start-Haltestellen. 
/// Die Spalten entsprechen den IDs der Ziel-Haltestellen ('col').
/// Die Zellen enthalten eine -1, falls keine passende Verbindungsfahrt existiert. Andernfalls
/// enthalten die Zellen die ID der betreffenden Ladestation (Haltestelle mit Ladestation). 
/// </summary>
class ChargingMatrix
{
public:
	ChargingMatrix(const CuProblem &problem, PlattformConfig plattform, bool performCheck);
	~ChargingMatrix();
	ChargingMatrix* getDevPtr();
	void copyToHost();

	/// <summary>
	/// Liefert die Haltestellen-Id der Ladestation, die auf dem Weg von der Haltestelle
	/// 'departure' zur Haltestelle 'destination' mit dem geringsten Zeitverlust zu erreichen ist.
	/// Die Zeit für den Weg von der Starthaltestelle zur Ladestation und anschliessend von der
	/// Ladestation zur Zielhaltestelle ist also bei der zurück gelieferten Ladestation am 
	/// geringsten.
	/// </summary>
	/// <param name="departure"></param>
	/// <param name="arrival"></param>
	/// <returns></returns>
	CU_HSTDEV StopId getFastestToReach(StopId departure, StopId destination) const;
	
private:
	void createChargingMatrixCpu(const CuProblem &problem);
	void createChargingMatrixGpu(const CuProblem &problem);

	/// <summary>
	/// Überprüft, ob die Matrix inhaltlich korrekt ist.
	/// </summary>
	/// <param name="problem"></param>
	/// <returns>True, wenn keine Fehler gefunden wurden. Andernfalls false.</returns>
	bool check(const CuProblem &problem);

	CuMatrix1<StopId> *_chargingMatrix;
	ChargingMatrix *_devicePtr;
};

