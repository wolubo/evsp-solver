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
	/// Die Werte können aus einem Textfile eingelesen werden. Es kann sich bspw. um eine Menge von Strassennamen handeln. 
	/// Sollten die Zeilen in Anführungszeichen eingeschlossen sein, so werden die Anführungszeichen entfernt.
	/// </summary>
	/// <param name="filename"></param>
	RandomValuePicker(const std::string& filename);

	~RandomValuePicker();

	/// <summary>
	/// Liefert einen zufällig aus der Liste ausgewählten Wert zurück.
	/// </summary>
	/// <param name="distinctPick">true: Der gelieferte Wert wird aus der Liste entfernt, kann also nicht nocheinmal geliefert werden.</param>
	/// <returns>Zufällig ausgewählter Wert oder null, falls keine Werte mehr verfügbar sind.</returns>
	std::string pickValue(bool distinctPick);

#pragma region private_member_variables
private:
	std::vector<std::string> _values;
	int _numOfRemainingLines = 0;
#pragma endregion
};
