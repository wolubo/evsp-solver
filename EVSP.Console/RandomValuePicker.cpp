#include <fstream>

#include "RandomValuePicker.h"


using namespace std;
//using namespace boost;

//#pragma warning(disable : 4996)

RandomValuePicker::RandomValuePicker(const string& filename)
{
	if (filename.length() == 0) throw std::invalid_argument("filename");

	ifstream file(filename);
	string str;
	while (std::getline(file, str))
	{
		_values.push_back(str);
	}

#pragma warning( push )  
#pragma warning( disable : 4267 ) // "Initialisierung": Konvertierung von XXX nach YYY, Datenverlust möglich"

	_numOfRemainingLines = _values.size();

	// Anführungszeichen entfernen.
	for (int i = 0; i< _values.size(); i++)
	{
		string s = _values[i];
		if (s[0] == '\"') s = s.substr(1);

		int lastChar = s.length() - 1;

		if (s[lastChar] == '\"') s = s.substr(0, lastChar);
		_values[i] = s;
	}

#pragma warning( pop )  
}


RandomValuePicker::~RandomValuePicker()
{
}


std::string RandomValuePicker::pickValue(bool distinctPick)
{
	if (_numOfRemainingLines < 1) return "<no more values to pick>";
	int r = rand() % _numOfRemainingLines;
	string retVal = _values[r];
	if (distinctPick)
	{
		_values[r] = _values[--_numOfRemainingLines]; // "Verbrauchten" Wert mit dem Inhalt der letzten Zeile überschreiben.
	}
	return retVal;
}
