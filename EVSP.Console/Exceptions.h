#pragma once

#include <string>
#include <exception>
#include <stdexcept>

#include "EVSP.BaseClasses/Typedefs.h"


class SyntaxErrorException : public std::runtime_error
{
public:
	SyntaxErrorException(int lineNo, const std::string& filename);
	SyntaxErrorException(int lineNo, const std::string& filename, const std::string& message);
	~SyntaxErrorException() {}
};


class SemanticErrorException : public std::runtime_error
{
public:
	SemanticErrorException(const std::string& filename, const std::string& message);
	~SemanticErrorException() {}
};


