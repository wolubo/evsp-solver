#pragma once

#include <string>

class RecentFile
{
public:
	RecentFile() = delete;
	RecentFile(std::string path, std::string filename, std::string extension);
	RecentFile(std::string path, std::string filename, std::string extension, std::string creationDate);
	~RecentFile();

	std::string getFilename() const;
	std::string getExtension() const;
	std::string getFullFilename() const;
	std::string getCreationDate() const;

private:
	std::string _extension;
	std::string _filename;
	std::string _path;
	std::string _creationDate;
};

