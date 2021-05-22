#include "Exceptions.h"

using namespace std;

SemanticErrorException::SemanticErrorException(const string &filename, const string &message)
	: runtime_error(message + " (Datei " + filename + ")")
{
}


SyntaxErrorException::SyntaxErrorException(int lineNo, const string& filename) 
	: runtime_error("Syntaxfehler in Zeile " + to_string(lineNo) + " der Datei " + filename + "!")
{
}


SyntaxErrorException::SyntaxErrorException(int lineNo, const string& filename, const string& message) 
	: runtime_error("Syntaxfehler in Zeile " + to_string(lineNo) + " der Datei " + filename + ": " + message)
{
}
