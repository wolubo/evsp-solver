#include <iomanip>
#include <sstream>

#include "Stopwatch.h"
#include "Configuration.h"


Stopwatch::Stopwatch() : start_t(0), stop_t(0)
{
	_isActive = Configuration::getConfiguration()->getStopwatch();
}


Stopwatch::~Stopwatch()
{
}


void Stopwatch::start()
{
	start_t = clock();
	stop_t = 0;
}


void Stopwatch::stop(string msg)
{
	stop_t = clock();
	if (msg.length() > 0 && _isActive) {
		cout << msg << elapsedSecondsAsStr(2) << " Sekunden" << endl;
	}
}


float Stopwatch::elapsedSeconds()
{
	clock_t temp;
	if (start_t > stop_t)
	{
		// Die Zeitmessung läuft noch. Gib den aktuellen "Zwischenstand" zurück.
		temp = clock();
	}
	else {
		temp = stop_t;
	}
	return (float)(temp - start_t) / CLOCKS_PER_SEC;
}


std::string Stopwatch::elapsedSecondsAsStr(int numOfDigits)
{
	stringstream stream;
	stream << fixed << setprecision(numOfDigits) << elapsedSeconds();
	return stream.str();
}

