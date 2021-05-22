#include "ResultLogger.h"
#include <iomanip>
#include <filesystem>
#include "EVSP.BaseClasses/Configuration.h"

namespace fs = std::filesystem;

ResultLogger::ResultLogger()
	: _logfile()
{
	try {
		std::shared_ptr<Configuration> config = Configuration::getConfiguration();
		string logfileDir;	// Verzeichnis, in dem das File erstellt werden soll.
		string filename;	// Dateiname, mit dem das File erstellt werden soll. 
		
		logfileDir = config->getWorkingDirectory() + "result";
		filename = "result.csv";

		if (fs::exists(logfileDir) || fs::create_directory(logfileDir)) {
			string completeFilename = logfileDir + "\\" + filename;
			if (fs::exists(completeFilename)) {
				// Suche eindeutigen Dateinamen.
				fs::path fnPath(filename);
				int sequenceNumber = 1;
				string distinctName;
				do {
					distinctName = logfileDir + "\\" + fnPath.stem().string() + "-" + std::to_string(sequenceNumber) + fnPath.extension().string();
					sequenceNumber++;
				} while (fs::exists(distinctName));
				completeFilename = distinctName;
			}
			_logfile.open(completeFilename);
			config->printConfig(_logfile);
			_logfile << endl;
			_logfile << "iteration;time;numOfVehicles;totalCost" << endl;
		}
	}
	catch (std::exception &ex) {
		cerr << "Es wird keine Ergebnis-Datei erzeugt (" << ex.what() << ")!" << endl;
	}
}


ResultLogger::~ResultLogger()
{
	if (_logfile.is_open()) {
		_logfile.close();
	}
}


void ResultLogger::addEntry(string caption, int iteration, float elapsedSec, AmountOfMoney totalCost, int numOfVehicles)
{
	if (caption.length() == 0) {
		caption = "Lösung";
	}
	cout << caption << " (Iteration= " << iteration << ", Zeit=" << setprecision(1) << elapsedSec << "): "
		<< "Fahrzeuganzahl=" << numOfVehicles << ", "
		<< "Gesamtkosten=" << (int)totalCost
		<< endl;
	if (_logfile.is_open()) {
		_logfile << iteration << ";" << elapsedSec << ";" << numOfVehicles << ";" << (int)totalCost << endl;
	}
}
