#pragma once
#include <memory>
#include "CuProblem.h"
#include "CuServiceTrips.h"
#include "CuEmptyTrips.h"
#include "CuStops.h"
#include "CuMatrix2.hpp"


class RandomGpu;




typedef CuMatrix1<ServiceTripId> RandomMatrix;
typedef CuMatrix2<bool, Max_VehicleTypeGroups, Max_VehicleTypeGroups> VehicleTypeGroupIntersection;


class MatrixCreator
{
public:

	/// <summary>
	/// Erzeugt eine CuMatrix2, deren Spalten mit Zufallszahlen in einem definierten Bereich gefüllt sind. 
	/// Jede Zahl kommt in einer Spalte nur einmal vor. Die Zufallszahlen liegen im Intervall [0,numOfCols[.
	/// </summary>
	/// <param name="random_dev">Quelle für Zufallszahlen im Device-Memory</param>
	static std::shared_ptr<RandomMatrix> createRandomMatrix(int numOfServiceTrips, int populationSize, shared_ptr<RandomGpu> rand);

	/// <summary>
	/// Generiert eine CuMatrix2, aus der sich ablesen lässt, ob sich zwei Fahrzeugtypgruppen überschneiden. Das ist dann der Fall,
	/// wenn es mindestens einen Fahrzeugtypen gibt, der in beiden Gruppen enthalten ist.
	/// Die Zeilen- und die Spaltenindizes entsprechen den Ids der Fahrzeugtypgruppen.
	/// Die Zellen enthalten 'true', falls sich die beiden Gruppen überschneiden. Andernfalls 'false'.
	/// </summary>
	/// <param name="problem"></param>				
	static shared_ptr<VehicleTypeGroupIntersection> createVehicleTypeGroupIntersection(std::shared_ptr<CuProblem> problem);

#ifdef _DEBUG
	static bool checkRandomMatrix(int numOfRows, shared_ptr<RandomMatrix> rndMatrix);

#endif

private:
	MatrixCreator() {}
	~MatrixCreator() {}
};

