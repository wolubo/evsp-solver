#pragma once

#include <string>
#include <map>
#include <vector>
#include "EVSP.BaseClasses/Typedefs.h"


class Block
{

public:
	Block();
	~Block();
	Block(const std::string& theName, const std::vector<std::string> &theColumns);

	void addLine(std::vector<std::string> splitted);

	/// <summary>
	/// Jeder Block hat einen eindeutigen Namen.
	/// </summary>
	std::string _blockName;

	/// <summary>
	/// Die Spalten-IDs, wie sie in der ersten Zeile des Blocks definiert sind.
	/// </summary>
	std::vector<std::string> _columns;

	/// <summary>
	/// Ein Block enthält eine Liste von Zeilen.
	/// Jede Zeile besteht aus einem Dictionary mit dem Key 'Spalten-ID' (--> 'Columns') und einem Value.
	/// </summary>
	std::vector<std::map<std::string, std::string>> _blockData;

};
