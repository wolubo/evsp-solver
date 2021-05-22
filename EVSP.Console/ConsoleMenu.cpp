#include <stdexcept>
#include <iostream>
#include <conio.h>

#include "ConsoleMenu.h"

using namespace std;
//using namespace boost;


ConsoleMenu::ConsoleMenu(string headline, vector<string> choices, bool exitOption)
	:_headline(headline), _choices(choices), _exitOption(exitOption)
{
	if (choices.size() >= _numOfKeys) throw invalid_argument("choices: Too many entries!");
}


ConsoleMenu::~ConsoleMenu()
{
}


int ConsoleMenu::displayMenu()
{
	cout << _headline << endl;

	for (int i = 0; i < _choices.size(); i++)
	{
		cout << _keys[i] << " " << _choices[i] << endl;
	}
	if (_exitOption) cout << "x Abbruch" << endl;

	char input;
	while (true)
	{
		input = _getch();
		if (input == 'x')
		{
			if (_exitOption) return -1;
		}
		else
		{
			for(int i = 0; i < _numOfKeys; i++)
			{
				if (input == _keys[i])
				{
					if (i < _choices.size())
						return i;
					else
						break;
				}
			}
		}
	}

}
