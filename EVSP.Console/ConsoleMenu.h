#pragma once

#include <string>
#include <vector>
#include "EVSP.BaseClasses/Typedefs.h"


/// <summary>
/// Zeigt innerhalb einer Konsolenanwendung ein konfigurierbares Menü an.
/// </summary>
class ConsoleMenu
{
public:
	ConsoleMenu(std::string headline, std::vector<std::string> choices, bool exitOption);
	~ConsoleMenu();

	/// <summary>
	/// Zeigt die in 'choices' übergebenen Menüpunkte an und liefert den Index des ausgewählten Menüpunktes oder -1 zurück, falls der Benutzer die Option 'Abbruch' gewählt hat.
	/// </summary>
	/// <param name="choices">Bezeichnungen der anzuzeigenden Menüpunkte.</param>
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






