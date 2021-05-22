#pragma once

#include <string>

#include "EVSP.Model/Problem.h"
#include "EVSP.BaseClasses/Typedefs.h"

class TextFileWriter
{
public:
	TextFileWriter();
	~TextFileWriter();

	void writeProblemDefinition(const std::shared_ptr<Problem> &problem, const std::string& filename);

private:
	void writeHeader_CHARGINGSYSTEM(std::ofstream &txtfile);
	void writeHeader_STOPPOINT(std::ofstream &txtfile);
	void writeHeader_LINE(std::ofstream &txtfile);
	void writeHeader_VEHICLETYPE(std::ofstream &txtfile);
	void writeHeader_VEHICLETYPEGROUP(std::ofstream &txtfile);
	void writeHeader_VEHTYPETOVEHTYPEGROUP(std::ofstream &txtfile);
	void writeHeader_VEHTYPECAPTOSTOPPOINT(std::ofstream &txtfile);
	void writeHeader_VEHTYPETOCHARGINGSTATION(std::ofstream &txtfile);
	void writeHeader_SERVICEJOURNEY(std::ofstream &txtfile);
	void writeHeader_DEADRUNTIME(std::ofstream &txtfile);
};

