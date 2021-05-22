#include "TextFileReader.h"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <exception>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

//#include "EVSP.BaseClasses/Configuration.h"
#include "EVSP.BaseClasses/Helper.h"
#include "Exceptions.h"

using namespace std;
using namespace boost;

//#pragma warning(disable : 4996)

int convert(const string& str, int defaultValue)
{
	int retVal = defaultValue;
	if (str.compare("-") != 0) retVal = stoi(str);
	return retVal;
}


PointInTime convertStringToPointInTime(std::string timeStr)
{
	std::vector<std::string> parts;
	boost::split(parts, timeStr, boost::is_any_of(":"));

	if (parts.size() != 4) throw std::invalid_argument("Zeitangabe mit unbekanntem Format: " + timeStr);

	int days = stoi(parts[0]);
	int hours = stoi(parts[1]);
	int minutes = stoi(parts[2]);
	int seconds = stoi(parts[3]);

	// Plausibilität prüfen.
	if (days > 999) throw std::invalid_argument("Zeitangabe mit mehr als 999 Tagen: " + timeStr);
	if (hours > 23) throw std::invalid_argument("Zeitangabe mit mehr als 23 Stunden: " + timeStr);
	if (minutes > 59) throw std::invalid_argument("Zeitangabe mit mehr als 59 Minuten: " + timeStr);
	if (seconds > 59) throw std::invalid_argument("Zeitangabe mit mehr als 59 Sekunden: " + timeStr);

	int numberOfSeconds = days * 86400     // Jeder Tag hat 86400 Sekunden.
		+ hours * 3600  // Jede Stunde hat 3600 Sekunden.
		+ minutes * 60
		+ seconds;

	return PointInTime(numberOfSeconds);
}


TextFileReader::TextFileReader()
	: _randomStreetNames(0)
{
	_config = Configuration::getConfiguration();
	if (_config->getUseRandomStreetnames()) {
		_randomStreetNames = make_shared<RandomValuePicker>(_config->getStreetnameFile());
	}
	_problem = make_shared<Problem>();
}


TextFileReader::~TextFileReader()
{
}


std::shared_ptr<Problem> TextFileReader::read(const string& filename, std::shared_ptr<Configuration> config)
{
	if (filename.length() == 0) throw invalid_argument("filename");

	_filename = filename;

	// Liste der vom Parser identifizierten Blöcke aus dem Textfile. 
	map<string, Block> blockList = readTextFile(filename, config);

	// Der Fileinhalt ist nun komplett im Dictionary '_blockList' enthalten. Nun können die Objekte erzeugt und in temporären Containern
	// gespeichert werden. Allerdings haben alle Objekte im Textfile eine ID, die nicht einfach übernommen werden kann. Sie wird lediglich 
	// verwendet, um die Relationen zwischen den Objekten wieder herzustellen. Danach wird sie verworfen und durch eine programmspezifische 
	// ID ersetzt.

		// Ladesysteme erzeugen. 
//	HandleBlock_CHARGINGSYSTEM(blockList["CHARGINGSYSTEM"]);

	// Haltestellen erzeugen.
	HandleBlock_STOPPOINT(blockList["STOPPOINT"]);

	// Linien erzeugen.
	HandleBlock_LINE(blockList["LINE"]);

	// Fahrzeugtypen erzeugen.
	HandleBlock_VEHICLETYPE(blockList["VEHICLETYPE"]);

	// Fahrzeugtypgruppen erzeugen.
	HandleBlock_VEHICLETYPEGROUP(blockList["VEHICLETYPEGROUP"]);

	// Fahrzeugtypen zu Fahrzeugtypgruppen zuordnen.
	HandleBlock_VEHTYPETOVEHTYPEGROUP(blockList["VEHTYPETOVEHTYPEGROUP"]);

	//"VEHTYPECAPTOSTOPPOINT"
	// VehTypeID;StoppointID;Min;Max
	{
		Block block = blockList["VEHTYPECAPTOSTOPPOINT"];
		if (block._blockData.size() > 0)
		{
			cerr << "Warnung: Die Einträge im Block VEHTYPECAPTOSTOPPOINT werden ignoriert!" << endl;
		}
	}

	// Fahrzeugtypen zu Ladestationen zuordnen.
	HandleBlock_VEHTYPETOCHARGINGSTATION(blockList["VEHTYPETOCHARGINGSTATION"]);

	// Linienfahrten erzeugen.
	HandleBlock_SERVICEJOURNEY(blockList["SERVICEJOURNEY"]);

	// Verbindungsfahrten erzeugen.
	HandleBlock_DEADRUNTIME(blockList["DEADRUNTIME"]);

	// Fahrzeugtypen in die Fahrzeugtypgruppen eintragen.
	vector<pair<string, string>>::iterator entry = _vehicleType2vehicleTypeGroup.begin();
	while (entry != _vehicleType2vehicleTypeGroup.end())
	{
		std::shared_ptr<VehicleType> vehicleType = _vehicleTypeMap[(*entry).first];
		std::shared_ptr<VehicleTypeGroup> vehicleTypeGroup = _vehicleTypeGroupMap[(*entry).second];
		vehicleTypeGroup->addVehicleType(vehicleType);
		entry++;
	}

	// Füge fehlende Ein-/Ausrückfahrten hinzu.
	if (_config->getCheckServiceTrips()) {
		_problem->checkServiceTrips(_config->isVerboseModeActive());
	}

	// Korrigiere Verbindungsfahrten mit ungültiger Distanz/Laufzeit.
	if (_config->getCheckEmptyTrips()) {
		_problem->checkEmptyTrips();
	}

	// Jedes Depot ist gleichzeitig auch Ladestation.
	for (int i = 0; i < _problem->getStops().size(); i++) {
		std::shared_ptr<Stop> busStop = _problem->getStops().at(i);
		if (busStop->isDepot()) {
			busStop->setChargingStation(true);
		}
	}

	// Prüfen: Gibt es Depots?
	bool hasDepots = false;
	for (int i = 0; i < _problem->getStops().size(); i++) {
		std::shared_ptr<Stop> busStop = _problem->getStops().at(i);
		if (busStop->isDepot()) hasDepots = true;
	}
	if (!hasDepots) throw SemanticErrorException("Die Datei enthält keine Depots!", filename);

	if (_config->getChargingStationDetectionMode() == ChargingStationDetectionMode::RANDOM_STOPS) {
		_problem->randomizeChargingStations(_config->getChargingStationRatio());
	}
	else if (_config->getChargingStationDetectionMode() == ChargingStationDetectionMode::NO_CHARGING_STATIONS) {
		_problem->solveForCombustionVehicles();
	}

	return _problem;
}


map<string, Block> TextFileReader::readTextFile(const string& filename, std::shared_ptr<Configuration> config)
{
	//map<string, Block> retVal = new map<string, Block>();
	map<string, Block> retVal;

	vector<string> lines;
	ifstream file(filename);

	if (!file) {
		string msg = "Die Datei " + filename + " existiert nicht!";
		throw runtime_error(msg);
	}

	string str;
	while (std::getline(file, str))
	{
		lines.push_back(str);
	}

	int lineNo = 1;
	Block currentBlock;
	bool isFirstBlock = true;
	vector<string> splitted;

	vector<string>::iterator iter = lines.begin();
	while (iter != lines.end())
	{
		string line = (*iter);
		trim(line);

		if (line.length() == 0)
		{
			// Leere Zeilen werden übersprungen.
		}
		else if (line[0] == '*')
		{
			// Kommentarzeile werden übersprungen.
		}
		else if (line[0] == '?')
		{
			// Konfigurationseinstellung
			split(splitted, line, is_any_of("="));
			if (splitted.size() != 2) throw SyntaxErrorException(lineNo, filename);
			if (!config->setKeyValue(splitted[0].substr(1), splitted[1])) throw SyntaxErrorException(lineNo, filename);
		}
		else if (line[0] == '$')
		{
			// Es beginnt ein neuer Block.
			if (!isFirstBlock)
			{
				// Der bislang eingelesene Block muss zunächst gespeichert werden, bevor mit einem neuen Block begonnen werden kann.
				retVal.insert(pair<string, Block>(currentBlock._blockName, currentBlock));
			}

			// Nun starte das Einlesen des neuen Blocks.
			split(splitted, line, is_any_of(":"));
			if (splitted.size() != 2) throw SyntaxErrorException(lineNo, filename);
			string blockName = splitted[0];
			to_upper(blockName); // In Großbuchstaben umwandeln.
			trim(blockName); // Whitespaces entfernen
			blockName = blockName.substr(1); // Dollar-Zeichen entfernen.

			split(splitted, splitted[1], is_any_of(";"));
			for (int i = 0; i < splitted.size(); i++) trim(splitted[i]); // Whitespaces entfernen.
			currentBlock = Block(blockName, splitted);
			isFirstBlock = false;
		}
		else
		{
			// Verarbeitung des aktuellen Blocks.
			split(splitted, line, is_any_of(";"));
			if (splitted.size() != currentBlock._columns.size()) {
				throw SyntaxErrorException(lineNo, filename, "Im Block " + currentBlock._blockName + " werden " +
					to_string(currentBlock._columns.size()) + " Spalten erwartet. Diese Zeile enthält aber " +
					to_string(splitted.size()) + " Spalten!");
			}
			for (int i = 0; i < splitted.size(); i++) trim(splitted[i]); // Whitespaces entfernen.
			currentBlock.addLine(splitted);
		}

		lineNo++;
		iter++;
	}

	// Den letzten eingelesenen Block speichern.
	retVal.insert(pair<string, Block>(currentBlock._blockName, currentBlock));

	return retVal;
}

void TextFileReader::HandleBlock_DEADRUNTIME(Block& block)
{
	//if (block == null) throw SemanticErrorException(_filename, "Die Datei enthält keine Verbingungsfahrten (DEADRUNTIME)!");

	bool warning = true;

	vector<map<string, string>> lines = block._blockData;
	for (int newId = 0; newId < lines.size(); newId++) {
		map<string, string> line = lines[newId];

		// FromStopID;ToStopID;FromTime;ToTime;Distance;RunTime
		string fromStopID = line["FromStopID"];
		string toStopID = line["ToStopID"];
		DistanceInMeters distance(stoi(line["Distance"]));
		DurationInSeconds runTime(stoi(line["RunTime"]));
		string fromTime = line["FromTime"];
		string toTime = line["ToTime"];

		if (warning && (fromTime.length() > 0 || toTime.length() > 0))
		{
			cerr << "Warnung: Gültigkeitszeiträume für Verbindungsfahrten werden ignoriert (DEADRUNTIME: FromTime, oTime)!" << endl;
			warning = false;
		}

		std::shared_ptr<Stop> fromStop = _stopMap[fromStopID];
		std::shared_ptr<Stop> toStop = _stopMap[toStopID];

		_problem->addEmptyTrip(make_shared<EmptyTrip>(fromStop, toStop, distance, runTime, fromTime, toTime));
	}
}


void TextFileReader::HandleBlock_SERVICEJOURNEY(Block& block)
{
	//if (block == null) throw SemanticErrorException(_filename, "Die Datei enthält keine Fahrzeugtypgruppen (VEHICLETYPEGROUP)!");

	if (_config->getDepotDetectionMode() == DepotDetectionMode::NO_SERVICE_TRIP) {
		// Depots finden: Jede Haltestelle, die keine Bushaltestelle ist, muss ein Depot sein. Denn es kommen weder SF an noch gehen welche ab.
		cout << "Hinweis: Jede Haltestelle, an der Servicefahrten weder ankommen noch abgehen wird als Depot angesehen." << endl;
		for (int i = 0; i < _problem->getStops().size(); i++) {
			std::shared_ptr<Stop> busStop = _problem->getStops().at(i);
			busStop->setDepot(true); // Zunächst alle auf true. Später werden alle, von denen Servicefahrten abgehen/ankommen auf false gesetzt.
		}
	}

	bool minAheadTimeWarning = true;
	bool maxShiftBackwardSecondsWarning = true;
	bool maxShiftForwardSecondsWarning = true;
	bool minLayoverTimeWarning = true;

	vector<map<string, string>> lines = block._blockData;
	for (int newId = 0; newId < lines.size(); newId++) {
		map<string, string> line = lines[newId];

		//ID;LineID;FromStopID;ToStopID;DepTime;ArrTime;MinAheadTime;MinLayoverTime;VehTypeGroupID;MaxShiftBackwardSeconds;
		//MaxShiftForwardSeconds;Distance
		string id = line["ID"];
		string lineID = line["LineID"];
		string fromStopID = line["FromStopID"];
		string toStopID = line["ToStopID"];
		string depTimeStr = line["DepTime"];
		string arrTimeStr = line["ArrTime"];
		int minAheadTime = stoi(line["MinAheadTime"]);
		int minLayoverTime = stoi(line["MinLayoverTime"]);
		string vehTypeGroupID = line["VehTypeGroupID"];
		int maxShiftBackwardSeconds = stoi(line["MaxShiftBackwardSeconds"]);
		int maxShiftForwardSeconds = stoi(line["MaxShiftForwardSeconds"]);
		DistanceInMeters distance(stoi(line["Distance"]));

		if (minAheadTimeWarning && minAheadTime != 0)
		{
			cerr << "Warnung: Bei Servicefahrten wird das Attribut MinAheadTime ignoriert (SERVICEJOURNEY)!" << endl;
			minAheadTimeWarning = false;
		}

		if (maxShiftBackwardSecondsWarning && maxShiftBackwardSeconds != 0)
		{
			cerr << "Warnung: Bei Servicefahrten wird das Attribut MaxShiftBackwardSeconds ignoriert (SERVICEJOURNEY)!" << endl;
			maxShiftBackwardSecondsWarning = false;
		}

		if (maxShiftForwardSecondsWarning && maxShiftForwardSeconds != 0)
		{
			cerr << "Warnung: Bei Servicefahrten wird das Attribut MaxShiftForwardSeconds ignoriert (SERVICEJOURNEY)!" << endl;
			maxShiftForwardSecondsWarning = false;
		}

		if (minLayoverTimeWarning && minLayoverTime != 0)
		{
			cerr << "Warnung: Bei Servicefahrten wird das Attribut MinLayoverTime ignoriert (SERVICEJOURNEY)!" << endl;
			minLayoverTimeWarning = false;
		}

		std::shared_ptr<BusRoute> route = _busRouteMap[lineID];
		std::shared_ptr<Stop> fromStop = _stopMap[fromStopID];
		std::shared_ptr<Stop> toStop = _stopMap[toStopID];
		std::shared_ptr<VehicleTypeGroup> vehTypeGroup = _vehicleTypeGroupMap[vehTypeGroupID];

		if (_config->getDepotDetectionMode() == DepotDetectionMode::NO_SERVICE_TRIP) {
			// Von diesen Haltestellen gehen Servicefahrten ab bzw. es kommen SF an. Es können also keine Depots sein.
			fromStop->setDepot(false);
			toStop->setDepot(false);
		}

		try
		{
			PointInTime depTime = convertStringToPointInTime(depTimeStr);
			PointInTime arrTime = convertStringToPointInTime(arrTimeStr);
			_problem->addServiceTrip(make_shared<ServiceTrip>(id, route, fromStop, toStop, depTime, arrTime, vehTypeGroup, distance, line["MinAheadTime"], line["MinLayoverTime"], line["MaxShiftBackwardSeconds"], line["MaxShiftForwardSeconds"]));
		}
		catch (const std::exception &ex)
		{
			throw SemanticErrorException(_filename, string(ex.what()) + " (Block SERVICEJOURNEY)");
		}

	}
}


void TextFileReader::HandleBlock_VEHTYPETOCHARGINGSTATION(Block& block)
{
	//if (block == null) throw  SemanticErrorException(_filename, "Die Datei enthält keine Zuordnung von Fahrzeugtypen zu Ladestationen (VEHTYPETOCHARGINGSTATION)!");

	if (_config->getChargingStationDetectionMode() == ChargingStationDetectionMode::VEH_TYPE_ENTRY) {
		cout << "Hinweis: Jede Haltestelle mit einem Eintrag im Block VEHTYPETOCHARGINGSTATION wird zur Ladestation." << endl;
	}

	// Fahrzeugtypen zu Ladestationen zuordnen.

	vector<map<string, string>> lines = block._blockData;
	for (int newId = 0; newId < lines.size(); newId++) {
		map<string, string> line = lines[newId];

		// VehTypeID;StoppointID
		//string vehTypeID = line["VehTypeID"];
		string busstopID = line["StoppointID"];

		//std::shared_ptr<VehicleType> vehicleType = _vehicleTypeMap[vehTypeID];
		std::shared_ptr<Stop> busStop = _stopMap[busstopID];

		if (_config->getChargingStationDetectionMode() == ChargingStationDetectionMode::VEH_TYPE_ENTRY) {
			busStop->setChargingStation(true);
		}

		//_vehicleType2chargingStation.push_back(pair<std::shared_ptr<Stop>, int>(busStop, vehicleType));
	}
}


void TextFileReader::HandleBlock_VEHTYPETOVEHTYPEGROUP(Block& block)
{
	// Fahrzeugtypen zu Fahrzeugtypgruppen zuordnen.

	//if (block == null) throw  SemanticErrorException(_filename, "Die Datei enthält keine Zuordnung von Fahrzeugtypen zu Fahrzeugtypgruppen (VEHTYPETOVEHTYPEGROUP)!");

	vector<map<string, string>> lines = block._blockData;
	for (int newId = 0; newId < lines.size(); newId++) {
		map<string, string> line = lines[newId];

		// VehTypeID;VehTypeGroupID
		string vehTypeID = line["VehTypeID"];
		string vehTypeGroupID = line["VehTypeGroupID"];

		_vehicleType2vehicleTypeGroup.push_back(pair<string, string>(vehTypeID, vehTypeGroupID));
	}
}


void TextFileReader::HandleBlock_VEHICLETYPEGROUP(Block& block)
{
	//if (block == null) throw  SemanticErrorException(_filename, "Die Datei enthält keine Fahrzeugtypgruppen (VEHICLETYPEGROUP)!");

	vector<map<string, string>> lines = block._blockData;
	for (int newId = 0; newId < lines.size(); newId++) {
		map<string, string> line = lines[newId];

		// ID;Code;Name
		string id = line["ID"];
		string code = line["Code"];
		string name = line["Name"];

		std::shared_ptr<VehicleTypeGroup> vtg = make_shared<VehicleTypeGroup>(id, code, name);
		_vehicleTypeGroupMap.insert(pair<string, std::shared_ptr<VehicleTypeGroup>>(id, vtg));
		_problem->addVehicleTypeGroup(vtg);
	}
}


void TextFileReader::HandleBlock_VEHICLETYPE(Block& block)
{
	//if (block == null) throw  SemanticErrorException(_filename, "Die Datei enthält keine Fahrzeugtypen (VEHICLETYPE)!");

	vector<map<string, string>> lines = block._blockData;
	for (int newId = 0; newId < lines.size(); newId++) {
		map<string, string> line = lines[newId];

		//ID;Code;Name;VehCharacteristic;VehClass;CurbWeightKg;VehCost;KmCost;HourCost;Capacity;KilowattHour;
		//ConsumptionPerServiceJourneyKm;ConsumptionPerDeadheadKm;RechargingCost;SlowRechargingTime;FastRechargingTime;ChargingSystem
		string id = line["ID"];
		string code = line["Code"];
		string name = line["Name"];
		string vehCharacteristic = line["VehCharacteristic"];
		string vehClass = line["VehClass"];
		int curbWeightKg = convert(line["CurbWeightKg"], 0);
		AmountOfMoney vehCost(stoi(line["VehCost"]));
		AmountOfMoney kmCost(stoi(line["KmCost"]));
		AmountOfMoney hourCost(stoi(line["HourCost"]));
		int capacity = convert(line["Capacity"], 0);
		KilowattHour batteryCapacity(toFloat(line["BatteryCapacity"]));
		KilowattHour consumptionPerServiceJourneyKm(toFloat(line["ConsumptionPerServiceJourneyKm"]));
		KilowattHour consumptionPerDeadheadKm(toFloat(line["ConsumptionPerDeadheadKm"]));
		AmountOfMoney rechargingCost(stoi(line["RechargingCost"]));
		int slowRechargingTime = convert(line["SlowRechargingTime"], INT_MAX);
		int fastRechargingTime = convert(line["FastRechargingTime"], INT_MAX);
		DurationInSeconds rechargingTime(min(slowRechargingTime, fastRechargingTime));

		//vector<int> supportedChargingSystems;
		//vector<string> chargingSystemStr;
		//split(chargingSystemStr, line["ChargingSystem"], is_any_of(","));

		//vector<string>::iterator chargingSystem = chargingSystemStr.begin();
		//while (chargingSystem != chargingSystemStr.end())
		//{
		//	int cs = _chargingSystemMap[*chargingSystem];
		//	supportedChargingSystems.push_back(cs);
		//	chargingSystem++;
		//}

		std::shared_ptr<VehicleType> vt = make_shared<VehicleType>(id, code, name, vehCost, kmCost, hourCost, batteryCapacity,
			consumptionPerServiceJourneyKm, consumptionPerDeadheadKm, rechargingCost, rechargingTime, /*supportedChargingSystems, */vehCharacteristic, vehClass, line["CurbWeightKg"], line["Capacity"], line["SlowRechargingTime"], line["FastRechargingTime"]);
		_vehicleTypeMap.insert(pair<string, std::shared_ptr<VehicleType>>(id, vt));
		_problem->addVehicleType(vt);

	}
}


void TextFileReader::HandleBlock_LINE(Block &block)
{
	//if (block == null) throw  SemanticErrorException(_filename, "Die Datei enthält keine Linien ($LINE)!");

	vector<map<string, string>> lines = block._blockData;
	for (int newId = 0; newId < lines.size(); newId++) {
		map<string, string> line = lines[newId];

		// ID;Code;Name
		string id = line["ID"];
		string code = line["Code"];
		string name = line["Name"];

		std::shared_ptr<BusRoute> newRoute = make_shared<BusRoute>(id, code, name);
		_busRouteMap.insert(pair<string, std::shared_ptr<BusRoute>>(id, newRoute));
		_problem->addRoute(newRoute);
	}
}


void TextFileReader::HandleBlock_STOPPOINT(Block& block)
{
	//if (block == null) throw  SemanticErrorException(_filename, "Die Datei enthält keine Haltestellen ($STOPPOINT)!");

	if (_config->getDepotDetectionMode() == DepotDetectionMode::ONE_DIGIT_ID) {
		cout << "Hinweis: Jede Haltestelle mit einer einstelligen Id wird als Depot angesehen." << endl;
	}

	if (_randomStreetNames)
	{
		cerr << "Hinweis: Die Haltestellennamen wurden durch zufällig gewählte Strassennamen ersetzt." << endl;
	}

	bool chargingStation = false;
	if (_config->getChargingStationDetectionMode() == ChargingStationDetectionMode::VEH_CAPACITY_AT_STOPS) {
		cout << "Hinweis: Jede Haltestelle mit eine Fahrzeug-Ladekapazität (VehCapacityForCharging) grösser als 0 ist eine Ladestation." << endl;
	}

	bool anonymizedStoppointWarning = true;

	vector<map<string, string>> lines = block._blockData;
	for (int newId = 0; newId < lines.size(); newId++) {
		map<string, string> line = lines[newId];

		// ID;Code;Name;VehCapacityForCharging
		string id = line["ID"];
		string code = line["Code"];
		string name = line["Name"];
		int vehCapacityForCharging = stoi(line["VehCapacityForCharging"]);

		if (anonymizedStoppointWarning)
		{
			bool anonymized = id.compare(name) == 0;
			if (anonymized)
			{
				cerr << "Hinweis: Die Datei scheint anonymisierte Haltestellen zu enthalten." << endl;
				anonymizedStoppointWarning = false;
			}
		}

		if (_randomStreetNames)
		{
			// Setze einen zufällig ausgewählten Namen für die Haltestelle.
			name = _randomStreetNames->pickValue(true);
		}

		chargingStation = false;
		if (_config->getChargingStationDetectionMode() == ChargingStationDetectionMode::VEH_CAPACITY_AT_STOPS) {
			chargingStation = (vehCapacityForCharging > 0);
		}

		std::shared_ptr<Stop> stop = make_shared<Stop>(id, code, name, chargingStation, line["VehCapacityForCharging"]);
		_stopMap.insert(pair<string, std::shared_ptr<Stop>>(id, stop));
		_problem->addStop(stop);

		if (_config->getDepotDetectionMode() == DepotDetectionMode::ONE_DIGIT_ID) {
			if (id.length() == 1) {
				stop->setDepot(true);
			}
		}
	}
}


//void TextFileReader::HandleBlock_CHARGINGSYSTEM(Block& block)
//{
//	//if (block == null) throw SemanticErrorException(_filename, "Die Datei enthält keine Haltestellen ($STOPPOINT)!");
//
//	// Ladesysteme erzeugen.
//
//	vector<map<string, string>> lines = block._blockData;
//	for (int newId = 0; newId < lines.size(); newId++) {
//		map<string, string> line = lines[newId];
//
//		// ID;Name
//		string id = line["ID"];
//		string name = line["Name"];
//		_chargingSystemMap.insert(pair<string, int>(id, newId));
//		_chargingSystems.push_back(make_shared<ChargingSystem>(id, name));
//
//	}
//}


