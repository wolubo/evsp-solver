#include "ConfigSettings.h"
#include <iostream>


using namespace std;


AcoParams::AcoParams()
	: populationSize(10000), printStats(true), dumpDecisionNet(false), dumpBestSolution(false), dumpWorstSolution(false), maxNumOfCirculations(700), 
	maxCirculationLength(50), chargeLevel(0.1f), performChecks(true), normalizeEdgeWeights(false), keepBestSolution(true)
{
}


void AcoParams::print(std::ostream &outStream)
{
	outStream << "Populationsgröße:             " << populationSize << endl;
	outStream << "Maximale Fahrzeuganzahl:      " << maxNumOfCirculations << endl;
	outStream << "Maximale Länge eines Umlaufs: " << maxCirculationLength << endl;
	outStream << "Grenzwert für Aufladungen:    " << chargeLevel << endl;

	outStream << "Kantengewichte normalisieren  ";
	if (normalizeEdgeWeights)
		outStream << "ja" << endl;
	else
		outStream << "nein" << endl;

	outStream << "Beste Lösung behalten         ";
	if (keepBestSolution)
		outStream << "ja" << endl;
	else
		outStream << "nein" << endl;

	outStream << "Entscheidungsnetz ausgeben:   ";
	if (dumpDecisionNet)
		outStream << "ja" << endl;
	else
		outStream << "nein" << endl;

	outStream << "Statistik ausgeben:           ";
	if (printStats)
		outStream << "ja" << endl;
	else
		outStream << "nein" << endl;

	outStream << "Beste Lösung ausgeben:        ";
	if (dumpBestSolution)
		outStream << "ja" << endl;
	else
		outStream << "nein" << endl;

	outStream << "Schlechteste Lösung ausgeben: ";
	if (dumpWorstSolution)
		outStream << "ja" << endl;
	else
		outStream << "nein" << endl;

	outStream << "Prüfungen durchführen:        ";
	if (performChecks)
		outStream << "Ja" << endl;
	else
		outStream << "Nein" << endl;
}


SaParams::SaParams()
	: populationSize(1000), maxCirculationLength(50), startTemperature(3000), minTemperature(1), crossoverRate(0.03f), crossoverUpperBound(INT_MAX),
	coolingRate(0.05f), greedyCreation(false), insertionRate(0.03f), insertionUpperBound(INT_MAX),
	deletionRate(0.1f), deletionsLowerBound(0), deletionsUpperBound(INT_MAX), printStats(false),
	performChecks(true), greedyInsertion(0.1f), greedyInsertionAfterDeletion(0.1f), crossoverChance(0.1f), insertionChance(0.1f), circCostDeletionChance(0.1f), numOfServiceTripsDeletionChance(0.1f), randomDeletionChance(0.1f)
{
	if (minTemperature <= 0) minTemperature = 1;
	if (startTemperature < minTemperature) startTemperature = minTemperature;

	if (coolingRate < 0.0f) coolingRate = 0.0f;
	if (coolingRate > 1.0f) coolingRate = 1.0f;

	if (crossoverRate < 0.0f) crossoverRate = 0.0f;
	if (crossoverRate > 1.0f) crossoverRate = 1.0f;
	if (crossoverUpperBound < 0) crossoverUpperBound = 0;

	if (insertionRate < 0.0f) insertionRate = 0.0f;
	if (insertionRate > 1.0f) insertionRate = 1.0f;
	if (insertionUpperBound < 0) insertionUpperBound = 0;

	if (deletionRate < 0.0f) deletionRate = 0.0f;
	if (deletionRate > 1.0f) deletionRate = 1.0f;

	if (deletionsLowerBound < 0) deletionsLowerBound = 0;
	if (deletionsUpperBound < deletionsLowerBound) deletionsUpperBound = deletionsLowerBound;
	if (deletionsLowerBound > deletionsUpperBound) deletionsLowerBound = deletionsUpperBound;
}


void SaParams::print(std::ostream &outStream)
{
	outStream << "Populationsgröße:             " << populationSize << endl;
	outStream << "Maximale Länge eines Umlaufs: " << maxCirculationLength << endl;
	outStream << "Mindesttemperatur:            " << minTemperature << endl;
	outStream << "Starttemperatur:              " << startTemperature << endl;
	outStream << "Abkühlungsrate:               " << coolingRate << endl;
	outStream << "Crossover-Rate:               " << crossoverRate * 100.0f << "%" << endl;
	outStream << "Crossover-Limit:              " << crossoverUpperBound << endl;

	outStream << "Erzeugung der Startlösung:    ";
	if (greedyCreation)
		outStream << "Greedy" << endl;
	else
		outStream << "Random" << endl;

	outStream << "Insertion-Operator:           ";
	if (greedyInsertion)
		outStream << "Greedy" << endl;
	else
		outStream << "Random" << endl;

	outStream << "Insertion-Operator Rate:      " << insertionRate * 100.0f << "%" << endl;
	outStream << "Insertion-Operator Limit:     " << insertionUpperBound << endl;

	outStream << "Deletion-Operator Rate:       " << deletionRate  * 100.0f << "%" << endl;
	outStream << "Deletion-Operator Limit:      " << deletionsLowerBound << endl;
	outStream << "Deletion-Operator Limit:      " << deletionsUpperBound << endl;

	outStream << "Operator-Wahrscheinlichkeit:" << endl;
	outStream << "  Crossover:                  " << crossoverChance  * 100.0f << "%" << endl;
	outStream << "  Insertion:                  " << insertionChance  * 100.0f << "%" << endl;
	outStream << "  Deletion (Kostenverhältn.): " << circCostDeletionChance  * 100.0f << "%" << endl;
	outStream << "  Deletion (Umlauflänge):     " << numOfServiceTripsDeletionChance  * 100.0f << "%" << endl;
	outStream << "  Deletion (zufällig):        " << randomDeletionChance  * 100.0f << "%" << endl;

	outStream << "Wahrscheinlichkeit für Greedy-Insertion:" << endl;
	outStream << "  nach Deletion-Operator:     " << greedyInsertionAfterDeletion  * 100.0f << "%" << endl;
	outStream << "  nach Insertion-Operator:    " << greedyInsertion  * 100.0f << "%" << endl;

	outStream << "Statistische Werte ausgeben:  ";
	if (printStats)
		outStream << "Ja" << endl;
	else
		outStream << "Nein" << endl;

	outStream << "Prüfungen durchführen:        ";
	if (performChecks)
		outStream << "Ja" << endl;
	else
		outStream << "Nein" << endl;
}



AcoQualifiers::AcoQualifiers()
	: fading(0.05f), totalCostQualifier(0.1f), numOfVehiclesQualifier(0.1f), circCostRatioQualifier(1.0f),
	/*strengthenTheBestSolution(1.0f), weakenTheBestSolution(1.0f),*/ weakenAllBadSolutions(1.0f), initialWeight(0.0f)
{
}


void AcoQualifiers::print(std::ostream &outStream) const
{
	outStream << "Faktoren für die Berechnung der Kantengewichte:" << endl;
	outStream << "   Initialgewicht:                                " << initialWeight << endl;
	outStream << "   Verblassen alter Kantengewichte (fade factor): " << fading << endl;
	outStream << "   Gewichtung der Gesamtkosten:                   " << totalCostQualifier << endl;
	outStream << "   Gewichtung der Fahrzeuganzahl:                 " << numOfVehiclesQualifier << endl;
	outStream << "   Gewichtung des Brutto-Netto-Kostenverh.:       " << circCostRatioQualifier << endl;
	//outStream << "   Verstärkung der besten Lösung:                 " << strengthenTheBestSolution << endl;
	//outStream << "   Abschwächen der besten Lösung:                 " << weakenTheBestSolution << endl;
	outStream << "   Abschwächungsfaktor für schlechte Lösungen:    " << weakenAllBadSolutions << endl;
}

