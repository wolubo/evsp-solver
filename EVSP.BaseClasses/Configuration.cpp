#include "Configuration.h"

#include <thread>
#include <stdexcept>
#include <iostream>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

using namespace std;
using namespace boost;

std::shared_ptr<Configuration> Configuration::_configuration = 0;
string Configuration::Filename = "config.ini";

std::shared_ptr<Configuration> Configuration::createConfiguration(std::string workingDirectory, bool cudaAvailability)
{
	if (!_configuration) {
		_configuration = std::make_shared<Configuration>(workingDirectory);
		if (_configuration->_plattform == GPU && !cudaAvailability) {
			_configuration->_plattform = CPU;
		}
	}
	return _configuration;
}

std::shared_ptr<Configuration> Configuration::getConfiguration()
{
	return _configuration;
}


Configuration::Configuration(std::string workingDirectory)
	: _workingDirectory(workingDirectory + "\\"),
	_checkEmptyTrips(false),
	_checkServiceTrips(false),
	_solver(SolverStrategyConfig::NO_SOLVER),
	_useRandomStreetnames(false),
	_depotDetectionMode(DepotDetectionMode::NO_SERVICE_TRIP),
	_chargingStationDetectionMode(ChargingStationDetectionMode::RANDOM_STOPS),
	_plattform(PlattformConfig::CPU),
	_chargingStationRatio(10),
	_terminateOnKeystroke(false),
	_terminateAfterTime(1000.0f),
	_terminateAfterRounds(1000),
	_performChecks(true),
	_maxNumOfThreads(INT_MAX),
	_minNumOfThreads(1),
	_threadFactor(2),
	_stopwatch(true),
	_verbose(false),
	_acoParams(),
	_saParams()
{
	if (!load()) {
		// Das Config-File kann nicht geladen werden. Default-Konfiguration erzeugen.
		string dataDir = _workingDirectory + "data\\";
		_recentFiles.push_back(make_shared<RecentFile>(dataDir, "Lackner_Set1", "txt", "01.01.2016"));
		_recentFiles.push_back(make_shared<RecentFile>(dataDir, "sample_real_867_SF_207_stoppoints", "txt", "01.01.2016"));
		_recentFiles.push_back(make_shared<RecentFile>(dataDir, "sample_real_2633_SF_67_stoppoints", "txt", "01.01.2016"));
		_recentFiles.push_back(make_shared<RecentFile>(dataDir, "sample_real_10710_SF_140_stoppoints", "txt", "01.01.2016"));

		_streetnameFile = dataDir + "strassen_osm.txt";
	}
}


Configuration::~Configuration()
{
	save();
}


bool Configuration::load()
{
	ifstream file(Filename);
	if (!file) return false;
	string line;
	vector<string> splitted;
	while (std::getline(file, line))
	{
		if (line.length() > 0 && line[0] != '*') // Kommentarzeilen beginnen mit einem * und werden einfach übersprungen.
		{
			split(splitted, line, is_any_of("="));
			if (splitted.size() != 2) {
				cerr << "Fehler in der Config-Datei!" << endl;
				return false;
			}
			string key = splitted[0];
			string value = splitted[1];
			if (!setKeyValue(key, value)) {
				cerr << "Fehler in der Config-Datei! Unbekannter Wert!" << endl;
				return false;
			}
		}
	}
	return true;
}


void Configuration::save()
{
}


bool Configuration::setKeyValue(map<string, string> &keyValues)
{
	map<string, string>::const_iterator iter = keyValues.begin();
	while (iter != keyValues.end()) {
		if (!setKeyValue((*iter).first, (*iter).second)) return false;
		iter++;
	}
	return true;
}

bool Configuration::setKeyValue(std::string key, std::string value)
{
	trim(key); // Whitespaces entfernen
	trim(value); // Whitespaces entfernen

	if (key.compare("WorkingDirectory") == 0) {
		_workingDirectory = value + "\\";
		return true;
	}

	if (key.compare("inputfile") == 0) {
		// Ignorieren.
		return true;
	}

	if (key.compare("use_random_streetnames") == 0) {
		to_lower(value);
		_useRandomStreetnames = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("depot_detection_mode") == 0) {
		to_lower(value);

		if (value.compare("no_service_trip") == 0) {
			_depotDetectionMode = DepotDetectionMode::NO_SERVICE_TRIP;
		}
		else if (value.compare("one_digit_id") == 0) {
			_depotDetectionMode = DepotDetectionMode::ONE_DIGIT_ID;
		}
		else {
			return false;
		}

		return true;
	}

	if (key.compare("charging_station_detection_mode") == 0) {
		to_lower(value);

		if (value.compare("no_charging_stations") == 0) {
			_chargingStationDetectionMode = ChargingStationDetectionMode::NO_CHARGING_STATIONS;
		}
		else if (value.compare("random_stops") == 0) {
			_chargingStationDetectionMode = ChargingStationDetectionMode::RANDOM_STOPS;
		}
		else if (value.compare("veh_capacity_at_stops") == 0) {
			_chargingStationDetectionMode = ChargingStationDetectionMode::VEH_CAPACITY_AT_STOPS;
		}
		else if (value.compare("veh_type_entry") == 0) {
			_chargingStationDetectionMode = ChargingStationDetectionMode::VEH_TYPE_ENTRY;
		}
		else {
			return false;
		}

		return true;
	}

	if (key.compare("charging_station_ratio") == 0) {
		int v = stoi(value);
		if (v >= 0 && v <= 100) {
			_chargingStationRatio = v;
			return true;
		}
		return true;
	}

	if (key.compare("MaxNumOfThreads") == 0) {
		int v = stoi(value);
		_maxNumOfThreads = v;
		return true;
	}

	if (key.compare("MinNumOfThreads") == 0) {
		int v = stoi(value);
		_minNumOfThreads = v;
		return true;
	}

	if (key.compare("ThreadFactor") == 0) {
		int v = stoi(value);
		_threadFactor = v;
		return true;
	}

	if (key.compare("stopwatch") == 0) {
		to_lower(value);
		_stopwatch = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("verbose") == 0) {
		to_lower(value);
		_verbose = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("solver") == 0) {
		to_lower(value);
		if (value.compare("aco") == 0)
			_solver = SolverStrategyConfig::ACO;
		else if (value.compare("sa") == 0)
			_solver = SolverStrategyConfig::SA;
		else
			return false;
		return true;
	}

	if (key.compare("plattform") == 0) {
		to_lower(value);
		if (value.compare("cpu") == 0)
			_plattform = PlattformConfig::CPU;
		else if (value.compare("gpu") == 0)
			_plattform = PlattformConfig::GPU;
		else
			return false;
		return true;
	}

	if (key.compare("streetfile") == 0) {
		_streetnameFile = _workingDirectory + value;
		return true;
	}

	if (key.compare("check_emtpy_trips") == 0) {
		to_lower(value);
		_checkEmptyTrips = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("check_service_trips") == 0) {
		to_lower(value);
		_checkServiceTrips = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("terminate_on_keystroke") == 0) {
		to_lower(value);
		_terminateOnKeystroke = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("terminate_after_time") == 0) {
		_terminateAfterTime = stof(value);
		return true;
	}

	if (key.compare("terminate_after_rounds") == 0) {
		_terminateAfterRounds = stoi(value);
		return true;
	}

	if (key.compare("recent_file") == 0) {
		vector<string> entry;
		split(entry, value, is_any_of("|"));
		if (entry.size() != 4) return false;
		trim(entry[0]); // Whitespaces entfernen
		trim(entry[1]);
		trim(entry[2]);
		trim(entry[3]);
		string path = _workingDirectory + entry[0];
		_recentFiles.push_back(make_shared<RecentFile>(path, entry[1], entry[2], entry[3]));
		return true;
	}

	if (key.compare("aco.print_stats") == 0) {
		to_lower(value);
		_acoParams.printStats = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("sa.print_stats") == 0) {
		to_lower(value);
		_saParams.printStats = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("perform_checks") == 0) {
		to_lower(value);
		_performChecks = (value.compare("true") == 0);
		_saParams.performChecks = _performChecks;
		_acoParams.performChecks = _performChecks;
		return true;
	}

	if (key.compare("aco.normalizeEdgeWeights") == 0) {
		to_lower(value);
		_acoParams.normalizeEdgeWeights = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("aco.keepBestSolution") == 0) {
		to_lower(value);
		_acoParams.keepBestSolution = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("aco.dump_decision_net") == 0) {
		to_lower(value);
		_acoParams.dumpDecisionNet = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("aco.dump_best_solution") == 0) {
		to_lower(value);
		_acoParams.dumpBestSolution = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("aco.dump_worst_solution") == 0) {
		to_lower(value);
		_acoParams.dumpWorstSolution = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("aco.population_size") == 0) {
		_acoParams.populationSize = stoi(value);
		return true;
	}

	if (key.compare("aco.max_circulation_length") == 0) {
		int v = stoi(value);
		_acoParams.maxCirculationLength = v;
		return true;
	}

	if (key.compare("aco.max_num_of_circulations") == 0) {
		int v = stoi(value);
		_acoParams.maxNumOfCirculations = v;
		return true;
	}

	if (key.compare("AcoQualifiers.InitialWeight") == 0) {
		_acoQualifiers.initialWeight = stof(value);
		return true;
	}

	if (key.compare("AcoQualifiers.fading") == 0) {
		_acoQualifiers.fading = stof(value);
		return true;
	}

	if (key.compare("AcoQualifiers.TotalCostQualifier") == 0) {
		_acoQualifiers.totalCostQualifier = stof(value);
		return true;
	}

	if (key.compare("AcoQualifiers.NumberOfVehiclesQualifier") == 0) {
		_acoQualifiers.numOfVehiclesQualifier = stof(value);
		return true;
	}

	if (key.compare("AcoQualifiers.CircCostRatioQualifier") == 0) {
		_acoQualifiers.circCostRatioQualifier = stof(value);
		return true;
	}

	//if (key.compare("AcoQualifiers.StrengthenTheBestSolution") == 0) {
	//	_acoQualifiers.strengthenTheBestSolution = stof(value);
	//	return true;
	//}

	//if (key.compare("AcoQualifiers.WeakenTheBestSolution") == 0) {
	//	_acoQualifiers.weakenTheBestSolution = stof(value);
	//	return true;
	//}

	if (key.compare("AcoQualifiers.WeakenAllBadSolutions") == 0) {
		_acoQualifiers.weakenAllBadSolutions = stof(value);
		return true;
	}

	if (key.compare("sa.population_size") == 0) {
		_saParams.populationSize = stoi(value);
		return true;
	}

	if (key.compare("sa.max_circulation_length") == 0) {
		int v = stoi(value);
		_saParams.maxCirculationLength = v;
		return true;
	}

	if (key.compare("SA.Params.StartTemperature") == 0) {
		_saParams.startTemperature = stof(value);
		return true;
	}

	if (key.compare("SA.Params.MinTemperature") == 0) {
		_saParams.minTemperature = stof(value);
		return true;
	}

	if (key.compare("SA.Params.CrossoverRate") == 0) {
		_saParams.crossoverRate = stof(value);
		return true;
	}

	if (key.compare("SA.Params.CrossoverUpperBound") == 0) {
		_saParams.crossoverUpperBound = stoi(value);
		return true;
	}

	if (key.compare("SA.Params.CoolingRate") == 0) {
		_saParams.coolingRate = stof(value);
		return true;
	}

	if (key.compare("SA.Params.GreedyCreation") == 0) {
		to_lower(value);
		_saParams.greedyCreation = (value.compare("true") == 0);
		return true;
	}

	if (key.compare("SA.Params.GreedyInsertion") == 0) {
		_saParams.greedyInsertion = stof(value);
		return true;
	}

	if (key.compare("SA.Params.InsertionRate") == 0) {
		_saParams.insertionRate = stof(value);
		return true;
	}

	if (key.compare("SA.Params.InsertionUpperBound") == 0) {
		_saParams.insertionUpperBound = stoi(value);
		return true;
	}

	if (key.compare("SA.Params.DeletionRate") == 0) {
		_saParams.deletionRate = stof(value);
		return true;
	}

	if (key.compare("SA.Params.DeletionsLowerBound") == 0) {
		_saParams.deletionsLowerBound = stoi(value);
		return true;
	}

	if (key.compare("SA.Params.DeletionsUpperBound") == 0) {
		_saParams.deletionsUpperBound = stoi(value);
		return true;
	}

	if (key.compare("SA.Params.GreedyInsertionAfterDeletion") == 0) {
		_saParams.greedyInsertionAfterDeletion = stof(value);
		return true;
	}

	if (key.compare("SA.Params.CrossoverChance") == 0) {
		_saParams.crossoverChance = stof(value);
		return true;
	}
	if (key.compare("SA.Params.InsertionChance") == 0) {
		_saParams.insertionChance = stof(value);
		return true;
	}
	if (key.compare("SA.Params.CircCostDeletionChance") == 0) {
		_saParams.circCostDeletionChance = stof(value);
		return true;
	}
	if (key.compare("SA.Params.NumOfServiceTripsDeletionChance") == 0) {
		_saParams.numOfServiceTripsDeletionChance = stof(value);
		return true;
	}
	if (key.compare("SA.Params.RandomDeletionChance") == 0) {
		_saParams.randomDeletionChance = stof(value);
		return true;
	}

	cerr << "Unbekannter Eintrag in der Konfiguration: " << key << endl;
	return false;
}


int Configuration::getNumOfThreads() const
{
	int numOfThreads = 1;
	if (!_verbose) {
		numOfThreads = thread::hardware_concurrency() * _threadFactor;
		if (numOfThreads < _minNumOfThreads) numOfThreads = _minNumOfThreads;
		if (numOfThreads > _maxNumOfThreads) numOfThreads = _maxNumOfThreads;
	}
	return numOfThreads;
}


AcoQualifiers Configuration::getQualifiers() const
{
	return _acoQualifiers;
}



void Configuration::printConfig(std::ostream &outStream)
{
	outStream << "Plattform:                       ";
	switch (_plattform) {
	case PlattformConfig::GPU:
		outStream << "GPU";
		break;
	case PlattformConfig::CPU:
		outStream << "CPU (" << thread::hardware_concurrency() << " Kerne, max. " << getNumOfThreads() << " Threads)";
		break;
	default:
		outStream << "<unbekannt>";
	}
	outStream << endl;

	outStream << "Haltestellennamen:               ";
	if (_useRandomStreetnames) {
		outStream << "zufällig" << endl;
	}
	else {
		outStream << "aus Eingabedatei" << endl;
	}

	outStream << "Prüfe Servicefahrten:            ";
	if (_checkServiceTrips) {
		outStream << "ja" << endl;
	}
	else {
		outStream << "nein" << endl;
	}

	outStream << "Prüfe Leerfahrten:               ";
	if (_checkEmptyTrips) {
		outStream << "ja" << endl;
	}
	else {
		outStream << "nein" << endl;
	}

	outStream << "Depots erkennen:                 ";
	switch (_depotDetectionMode) {
	case DepotDetectionMode::NO_SERVICE_TRIP:
		outStream << "Haltestelle ohne Servicefahrten";
		break;
	case DepotDetectionMode::ONE_DIGIT_ID:
		outStream << "einstellige Id";
		break;
	default:
		outStream << "<unbekannt>";
	}
	outStream << endl;

	outStream << "Ladestation erkennen:            ";
	switch (_chargingStationDetectionMode) {
	case ChargingStationDetectionMode::NO_CHARGING_STATIONS:
		outStream << "keine (konventionelles VSP lösen)";
		break;
	case ChargingStationDetectionMode::RANDOM_STOPS:
		outStream << "zufällige Verteilung (" << _chargingStationRatio << "%)";
		break;
	case ChargingStationDetectionMode::VEH_CAPACITY_AT_STOPS:
		outStream << "Haltestelle mit Ladekapazität (VehCapacityForCharging)";
		break;
	case ChargingStationDetectionMode::VEH_TYPE_ENTRY:
		outStream << "Haltestelle mit Eintrag im Block VEHTYPETOCHARGINGSTATION";
		break;
	default:
		outStream << "<unbekannt>";
	}
	outStream << endl;

	outStream << "Lösungsverfahren:                ";
	switch (_solver) {
	case SolverStrategyConfig::ACO: {
		outStream << "Ant Colony Optimization" << endl;
		_acoParams.print(outStream);
		_acoQualifiers.print(outStream);
		break;
	}
	case SolverStrategyConfig::SA:
		outStream << "Simulated Annealing" << endl;
		_saParams.print(outStream);
		break;
	case SolverStrategyConfig::NO_SOLVER:
		outStream << "<nicht spezifiziert>" << endl;
		break;
	default:
		outStream << "<unbekannt>" << endl;
	}

	outStream << "Arbeitsverzeichnis: " << _workingDirectory << endl;

	if (_stopwatch) {
		outStream << "Es werden Laufzeiten gemessen und angezeigt." << endl;
	}
	else {
		outStream << "Laufzeiten werden nicht angezeigt." << endl;
	}

	if (_verbose) {
		outStream << "VERBOSE-MODE AKTIV!" << endl;
	}
}

