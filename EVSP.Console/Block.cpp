#include "Block.h"



using namespace std;
//using namespace boost;


Block::Block()
{
	_blockName = string();
	_columns   = vector<string>();
	_blockData = vector<map<string, string>>();
}


Block::Block(const string& theName, const vector<string> &theColumns)
{
	_blockName = theName;
	_columns   = theColumns;
	_blockData = vector<map<string, string>>();
}


Block::~Block()
{
}


void Block::addLine(vector<string> splitted)
{
	map<string, string> current;

	for (int i = 0; i < _columns.size(); i++)
	{
		if (_columns[i].length() > 0)
		{
			current.insert(pair<string, string>(_columns[i], splitted[i]));
		}
		else
		{
			if (splitted[i].length() > 0) throw  invalid_argument("Zeile enthält Daten, aber es ist keine Spalten-ID definiert!");
		}
	}
	_blockData.push_back(current);
}
