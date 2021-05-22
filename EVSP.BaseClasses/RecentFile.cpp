#include <ctime>

#include "RecentFile.h"


using namespace std;


RecentFile::RecentFile(std::string path, std::string filename, std::string extension)
	: _path(path), _filename(filename), _extension(extension)
{
	time_t t = time(0);
	struct tm *now = localtime(&t);
	_creationDate = to_string(now->tm_mday) + "." + to_string(now->tm_mon + 1) + "." + to_string(now->tm_year + 1900);
}


RecentFile::RecentFile(string path, string filename, string extension, string creationDate)
	: _path(path), _filename(filename), _extension(extension), _creationDate(creationDate)
{
}


RecentFile::~RecentFile()
{
}


std::string RecentFile::getFilename() const
{
	return _filename;
}


std::string RecentFile::getExtension() const
{
	return _extension;
}


string RecentFile::getFullFilename() const
{
	string retVal = "";
	if (_path.length() > 0) retVal = _path + "\\";
	retVal += _filename;
	if (_extension.length() > 0) retVal += "." + _extension;
	return retVal;
}


std::string RecentFile::getCreationDate() const
{
	return _creationDate;
}

