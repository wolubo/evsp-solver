#include <conio.h>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include <stdio.h>
#include <direct.h>
#include <iostream>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "EVSP.Model/ServiceTrip.h"

#include "EVSP.BaseClasses/Stopwatch.h"
#include "EVSP.BaseClasses/Configuration.h"
#include "EVSP.BaseClasses/RecentFile.h"
#include "ConsoleMenu.h"

#include "TextFileReader.h"
#include "../EVSP.CudaSolver/CuSolver.h"
#include "../EVSP.CudaSolver/SolverFabric.h"

#ifdef _DEBUG
#include "TextFileWriter.h"
#endif


string selectFile()
{
	std::shared_ptr<Configuration> config = Configuration::getConfiguration();

	vector<string> recentFilesMenu;

	vector<std::shared_ptr<RecentFile>> recentFiles = config->getRecentFiles();

	vector<std::shared_ptr<RecentFile>>::iterator iter = recentFiles.begin();
	while (iter != recentFiles.end()) {
		string s = (*iter)->getFilename() + "." + (*iter)->getExtension() + " (" + (*iter)->getCreationDate() + ")";
		recentFilesMenu.push_back(s);
		iter++;
	}

	ConsoleMenu menu("Datei laden:", recentFilesMenu, true);
	int choice = menu.displayMenu();

	if (choice < 0)
	{
		std::exit(0);
	}
	else if (choice < recentFilesMenu.size())
	{
		// Öffne eine der zuletzt geöffneten Dateien.
		return recentFiles[choice]->getFullFilename();
	}
	return "<error>";
}


map<string, string> handleCommandLineArguments(int argc, char* argv[])
{
	map<string, string> retVal;
	vector<string> entry;
	string key, val;

	for (int i = 1; i < argc; i++)
	{
		string arg = string(argv[i]);
		boost::split(entry, arg, boost::is_any_of("="));
		if (entry.size() == 2)
		{
			key = entry[0];
			val = entry[1];
			boost::trim(key);
			boost::to_lower(key);
			boost::trim(val);
			retVal.insert(pair<string, string>(key, val));
		}
	}
	return retVal;
}

std::shared_ptr<CuSolver> setupSolver(std::shared_ptr<Configuration> config, std::shared_ptr<Problem> problem);

// Wenn der Benutzer eine Taste drückt soll die Suche terminieren.
bool terminateSolver() {
	bool retVal = _kbhit() == 1;
	if (retVal) _getch(); // Tastendruck verarbeiten, damit er nicht im Buffer "liegen" bleibt.
	return retVal;
}


std::string GetCurrentWorkingDir(void) {
	char buff[FILENAME_MAX];
	_getcwd(buff, FILENAME_MAX);
	std::string current_working_dir(buff);
	return current_working_dir;
}


int main(int argc, char* argv[])
{
	try {

		//int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
		//tmpFlag &= ~_CRTDBG_CHECK_ALWAYS_DF;
		//_CrtSetDbgFlag(tmpFlag);

		std::locale::global(std::locale("German_germany"));

		bool cudaAvailability = CuSolver::setupCuda();

		// Kopiere den Pfad des Arbeitsverzeichnisses
		std::string wd = GetCurrentWorkingDir();

		std::shared_ptr<Configuration> config = Configuration::createConfiguration(wd, cudaAvailability);

		Stopwatch stopwatch;

		map<string, string> commandLineArguments = handleCommandLineArguments(argc, argv);

		string inputfile;
		{
			map<string, string>::iterator it = commandLineArguments.find("inputfile");
			if (it != commandLineArguments.end())
			{
				inputfile = commandLineArguments["inputfile"];
			}
			else
			{
				inputfile = selectFile();
				if (inputfile.length() == 0) std::exit(1);
			}
		}

		config->setKeyValue(commandLineArguments);

		stopwatch.start();

		cout << "Lade " << inputfile << "..." << endl;

		std::shared_ptr<Problem> problem;
		TextFileReader reader;
		problem = reader.read(inputfile, config);

#ifdef _DEBUG
		TextFileWriter writer;
		writer.writeProblemDefinition(problem, inputfile + ".test");
#endif

		cout << endl << "Konfiguration:" << endl;
		config->printConfig(cout);
		problem->printStatistic();

		string msg = inputfile + " geladen in ";
		stopwatch.stop(msg);

		std::shared_ptr<CuSolver> solver = setupSolver(config, problem);
		if (!solver) exit(-1);

		// Kriterien für das Terminieren des Solvers festlegen.
		solver->setMaxElapsedTime(config->getTerminateAfterTime());
		solver->setMaxNumOfRounds(config->getTerminateAfterRounds());
		if (config->getTerminateOnKeystroke()) {
			solver->setTerminationDelegate(terminateSolver);
		}

		cout << endl << "Lösungssuche..." << endl;
		stopwatch.start();

		solver->run();
		std::shared_ptr<CuSolution> solution = solver->getSolution();

		stopwatch.stop("Suche beendet nach ");

		//cout << endl << "Lösung:" << endl;
		//cout << solution->toString() << endl;

		// Sicherstellen, dass der allokierte Device-Speicher vor cudaDeviceReset() freigegeben wird.
		//cuProblem = 0; 
		//solver = 0;
		problem.reset();
		solver.reset();
		solution.reset();
		config.reset();

		if (cudaAvailability) {
			CuSolver::resetCuda();
		}

		cout << endl << "Taste drücken..." << endl;
		_getch();

	} 
	catch (const std::exception &err) {
		cerr << "FEHLER: " << err.what() << endl;
		return -1;
	}

 	return 0;
}


SolverStrategyConfig selectSolverStrategy(std::shared_ptr<Configuration> config)
{
	SolverStrategyConfig solverChoice = config->getSolver();
	if (solverChoice == SolverStrategyConfig::NO_SOLVER) {
		cout << endl;

		vector<string> solverMenu;
		solverMenu.push_back("Simulated Annealing");
		solverMenu.push_back("Ant Colony Optimization");

		ConsoleMenu menu("Lösungsstrategie wählen:", solverMenu, true);
		int choice = menu.displayMenu();

		if (choice < 0)
		{
			exit(0);
		}
		else if (choice == 0)
		{
			return SolverStrategyConfig::SA;
		}
		else if (choice == 1)
		{
			return SolverStrategyConfig::ACO;
		}
		if (solverChoice == SolverStrategyConfig::NO_SOLVER) exit(1);
	}
	return solverChoice;
}


PlattformConfig selectPlattform(std::shared_ptr<Configuration> config)
{
	PlattformConfig solverChoice = config->getPlattform();
	if (solverChoice == PlattformConfig::UNDEFINED) {

		vector<string> plattformMenu;
		plattformMenu.push_back("CPU");
		plattformMenu.push_back("GPU");

		ConsoleMenu menu("Plattform wählen:", plattformMenu, true);
		int choice = menu.displayMenu();

		if (choice < 0)
		{
			exit(0);
		}
		else if (choice == 0)
		{
			return PlattformConfig::CPU;
		}
		else if (choice == 1)
		{
			return PlattformConfig::GPU;
		}
	}
	return solverChoice;
}


std::shared_ptr<CuSolver> setupSolver(std::shared_ptr<Configuration> config, std::shared_ptr<Problem> problem)
{
	std::shared_ptr<CuSolver> solver;
	std::shared_ptr<SolverFabric> solverFabric = std::make_shared<SolverFabric>(config, problem);

	switch (selectSolverStrategy(config)) {
	case SolverStrategyConfig::SA:
	{
		switch (selectPlattform(config)) {
		case PlattformConfig::CPU:
			cout << "Lösung erfolgt auf der CPU durch Simulated Annealing." << endl;
			break;
		case PlattformConfig::GPU:
			cout << "Lösung erfolgt auf der GPU durch Simulated Annealing." << endl;
			break;
		default:
			cerr << "FEHLER: Unbekannte Plattform!" << endl;
			exit(-1);
		}
		solver = solverFabric->createSaSolver();
		break;
	}
	case SolverStrategyConfig::ACO:
	{
		switch (selectPlattform(config)) {
		case PlattformConfig::CPU:
			cout << "Lösung erfolgt auf der CPU durch Anwendung eines Ameisenalgorithmus." << endl;
			break;
		case PlattformConfig::GPU:
			cout << "Lösung erfolgt auf der GPU durch Anwendung eines Ameisenalgorithmus." << endl;
			break;
		default:
			cerr << "FEHLER: Unbekannte Plattform!" << endl;
			exit(-1);
		}
		solver = solverFabric->createAcoSolver();
		break;
	}
	default:
		cerr << "FEHLER: Unbekannte Lösungsstrategie!" << endl;
		exit(-1);
	}

	return solver;
}

