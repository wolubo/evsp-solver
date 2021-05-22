#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include "EVSP.BaseClasses/Typedefs.h"


class RandomValuePicker
{
public:

	RandomValuePicker() = delete;

	/// <summary>
	/// Die Werte k�nnen aus einem Textfile eingelesen werden. Es kann sich bspw. um eine Menge von Strassennamen handeln. 
	/// Sollten die Zeilen in Anf�hrungszeichen eingeschlossen sein, so werden die Anf�hrungszeichen entfernt.
	/// </summary>
	/// <param name="filename"></param>
	RandomValuePicker(const std::string& filename);

	~RandomValuePicker();

	/// <summary>
	/// Liefert einen zuf�llig aus der Liste ausgew�hlten Wert zur�ck.
	/// </summary>
	/// <param name="distinctPick">true: Der gelieferte Wert wird aus der Liste entfernt, kann also nicht nocheinmal geliefert werden.</param>
	/// <returns>Zuf�llig ausgew�hlter Wert oder null, falls keine Werte mehr verf�gbar sind.</returns>
	std::string pickValue(bool distinctPick);

#pragma region private_member_variables
private:
	std::vector<std::string> _values;
	int _numOfRemainingLines = 0;
#pragma endregion
};
