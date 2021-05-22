#pragma once

#include <time.h>
#include <string>
#include "Typedefs.h"

using namespace std;

class Stopwatch
{
public:
	Stopwatch();
	~Stopwatch();

	/// <summary>
	/// Startet die Zeitmessung. Beginnt die Messung bei 0 Sekunden.
	/// </summary>
	void start();

	/// <summary>
	/// Stopt die Zeitmessung und speichert die gemessene Zeit.
	/// </summary>
	void stop(string msg);

	/// <summary>
	/// Liefert die seit dem letzten Aufruf von 'start()' gemessene Zeit. Sofern zwischenzeitlich 'stop()' aufgerufen wurde wird 
	/// die gestoppte Zeit geliefert.
	/// </summary>
	float elapsedSeconds();

	/// <summary>
	/// Liefert die mit 'elapsedSeconds()' ermittelte Zeit als String.
	/// <param name="numOfDigits">Anzahl der Nachkommastellen.</param>
	/// </summary>
	std::string elapsedSecondsAsStr(int numOfDigits);

private:
	clock_t start_t, stop_t;
	bool _isActive;
};
