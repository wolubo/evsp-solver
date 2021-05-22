#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include "EVSP.BaseClasses/Typedefs.h"

using namespace std;

/// <summary>
/// Gibt alle anfallenden (Zwischen-) Ergebnisse auf der Konsole aus und schreibt sie zus�tzlich in ein CSV-File, 
/// sodass sie sp�ter mit Excel ausgewertet werden k�nnen.
/// </summary>
class ResultLogger
{
public:
	/// <summary>
	/// �ffnet ein File, in das alle Ergebnisse geschrieben werden. 
	/// </summary>
	ResultLogger();

	/// <summary>
	/// Schlie�t das Ergebnis-File.
	/// </summary>
	~ResultLogger();

	/// <summary>
	/// F�gt der Liste der (Zwischen-) Ergebnisse einen Eintrag hinzu.
	/// </summary>
	/// <param name="caption">String, der auf der Konsole am Anfang der Ergebnis-Zeile ausgegeben wird.</param>
	/// <param name="iteration">Iterations-Nummer</param>
	/// <param name="elapsedSec">Anzahl der bisher vergangenen Sekunden.</param>
	/// <param name="totalCost">Gesamtkosten</param>
	/// <param name="numOfVehicles">Anzahl der ben�tigten Fahrzeuge.</param>
	void addEntry(string caption, int iteration, float elapsedSec, AmountOfMoney totalCost, int numOfVehicles);

private:
	std::ofstream _logfile;
};

