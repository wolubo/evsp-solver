#pragma once

#include <string>
#include <vector>
#include "EVSP.BaseClasses/Typedefs.h"


/// <summary>
/// Zeigt innerhalb einer Konsolenanwendung ein konfigurierbares Men� an.
/// </summary>
class ConsoleMenu
{
public:
	ConsoleMenu(std::string headline, std::vector<std::string> choices, bool exitOption);
	~ConsoleMenu();

	/// <summary>
	/// Zeigt die in 'choices' �bergebenen Men�punkte an und liefert den Index des ausgew�hlten Men�punktes oder -1 zur�ck, falls der Benutzer die Option 'Abbruch' gew�hlt hat.
	/// </summary>
	/// <param name="choices">Bezeichnungen der anzuzeigenden Men�punkte.</param>
	/// <param name="exitOption">Falls true: Zeige die Option 'Abbruch' an.</param>
	/// <returns></returns>
	int displayMenu();

private:
	std::string _headline;
	std::vector<std::string> _choices;
	bool _exitOption;
	const char _keys[24] = { '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'x' };
	const unsigned __int8 _numOfKeys = 24;
};






