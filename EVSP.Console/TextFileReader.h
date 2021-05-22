#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>

#include "EVSP.Model/Problem.h"
#include "EVSP.Model/ChargingSystem.h"
#include "EVSP.Model/Stop.h"
#include "EVSP.Model/BusRoute.h"
#include "EVSP.Model/VehicleType.h"
#include "EVSP.Model/VehicleTypeGroup.h"
#include "EVSP.Model/EmptyTrip.h"
#include "EVSP.Model/ServiceTrip.h"

#include "Block.h"
#include "RandomValuePicker.h"
#include "EVSP.BaseClasses/Configuration.h"
#include "EVSP.BaseClasses/Typedefs.h"

using namespace std;


/// <summary>
/// Modell für die im Definition-File enthaltenen Daten.
/// </summary>
class TextFileReader
{
public:
	TextFileReader();
	~TextFileReader();

	/// <summary>
	/// Lädt ein Textfile.
	/// Fileformat:
	/// Die Files setzen sich aus Blöcken zusammen. Jeder Block startet mit einer Zeile, die mit einem Dollar-Zeichen ($) beginnt und 
	/// den Bezeichner des Blocks, einen Doppelpunkt und eine durch Semikola separierte Liste der im Block verwendeten Spaltenbezeichner enthält. 
	/// Die dann folgenden Zeilen enthalten die Daten des Blocks (ebenfalls durch Semikola separiert).
	/// Kommentarzeilen starten mit einem Asterix (*) in der ersten Spalte.
	/// 
	/// Beispiel: 
	/// * Haltestellen
	/// $STOPPOINT:ID;Code;Name;VehCapacityForCharging
	/// 8890;8890;8890;1000
	/// 8886;8886;8886;1000
	/// </summary>
	/// <param name="filename">Name und Pfad zum File, aus dem gelesen werden soll.</param>
	std::shared_ptr<Problem> read(const string& filename, std::shared_ptr<Configuration> config);


#pragma region private_member_functions
private:

	/// <summary>
	/// Liest den Inhalt eines Textfiles ein und teilt den Inhalt in Blöcke auf.
	/// </summary>
	/// <param name="filename"></param>
	/// <returns></returns>
	map<string, Block> readTextFile(const string& filename, std::shared_ptr<Configuration> config);

	void HandleBlock_DEADRUNTIME(Block& block);

	/// <summary>
	/// Verarbeitet den Block SERVICEJOURNEY, in dem alle Linienfahrten enthalten sind.
	/// Markiert ausserdem alle Bushaltestellen, bei denen Servicefahrten ankommen oder abgehen.
	/// </summary>
	void HandleBlock_SERVICEJOURNEY(Block& block);

	/// <summary>
	/// Verarbeitet den Block VEHTYPETOCHARGINGSTATION, in dem die Zuordnungen der Fahrzeugtypen zu Ladestationen enthalten sind.
	/// </summary>
	void HandleBlock_VEHTYPETOCHARGINGSTATION(Block& block);

	/// <summary>
	/// Verarbeitet den Block VEHTYPETOVEHTYPEGROUP, in dem die Zuordnungen der Fahrzeugtypen zu Fahrzeugtypgruppen enthalten sind.
	/// </summary>
	void HandleBlock_VEHTYPETOVEHTYPEGROUP(Block& block);

	/// <summary>
	/// Verarbeitet den Block VEHICLETYPEGROUP, in dem alle Fahrzeugtypgruppen enthalten sind.
	/// </summary>
	void HandleBlock_VEHICLETYPEGROUP(Block& block);

	/// <summary>
	/// Verarbeitet den Block VEHICLETYPE, in dem alle Fahrzeugtypen enthalten sind.
	/// </summary>
	void HandleBlock_VEHICLETYPE(Block& block);

	/// <summary>
	/// Verarbeitet den Block LINE, in dem alle Buslinien enthalten sind.
	/// </summary>
	void HandleBlock_LINE(Block &block);

	/// <summary>
	/// Verarbeitet den Block STOPPOINT, in dem alle Bushaltestellen enthalten sind.
	/// </summary>
	void HandleBlock_STOPPOINT(Block& block);

	/// <summary>
	/// Verarbeitet den Block CHARGINGSYSTEM, in dem alle Ladesysteme enthalten sind.
	/// </summary>
	/// <param name="Result">Return-Value des Parsers. Die eingelesenen Ladesysteme werden zu 'Result' hinzugefügt.</param>
	//void HandleBlock_CHARGINGSYSTEM(Block& block);

#pragma endregion

#pragma region private_member_variables
private:

	string _filename;
	std::shared_ptr<RandomValuePicker> _randomStreetNames;
	std::shared_ptr<Problem> _problem;
	std::shared_ptr<Configuration> _config;

	// Temporäres Verzeichnis der Ids aller Ladesysteme. Als Key wird die im Textfile definierte ID verwendet, um die Zuordnung 
	// der Objekte vornehmen zu können. Nachdem alle Zuordnungen erfolgt sind wird dieses Verzeichnis nicht mehr benötigt.
	map<string, int> _chargingSystemMap;

	// Temporäres Verzeichnis aller Haltestellen. Als Key wird die im Textfile definierte ID verwendet, um die Zuordnung der Objekte vornehmen zu 
	// können. Nachdem alle Zuordnungen erfolgt sind wird dieses Verzeichnis nicht mehr benötigt.
	map<string, std::shared_ptr<Stop>> _stopMap;

	// Temporäres Verzeichnis aller Buslinien. Als Key wird die im Textfile definierte ID verwendet, um die Zuordnung der Objekte vornehmen zu 
	// können. Nachdem alle Zuordnungen erfolgt sind wird dieses Verzeichnis nicht mehr benötigt.
	map<string, std::shared_ptr<BusRoute>> _busRouteMap;

	// Temporäres Verzeichnis aller Fahrzeugtypen. Als Key wird die im Textfile definierte ID verwendet, um die Zuordnung der Objekte vornehmen zu 
	// können. Nachdem alle Zuordnungen erfolgt sind wird dieses Verzeichnis nicht mehr benötigt.
	map<string, std::shared_ptr<VehicleType>> _vehicleTypeMap;

	// Temporäres Verzeichnis aller Fahrzeugtypgruppen. Als Key wird die im Textfile definierte ID verwendet, um die Zuordnung der Objekte vornehmen zu 
	// können. Nachdem alle Zuordnungen erfolgt sind wird dieses Verzeichnis nicht mehr benötigt.
	map<string, std::shared_ptr<VehicleTypeGroup>> _vehicleTypeGroupMap;

	///<summary>
	/// first: VehicleTypeGroup
	/// second: VehicleType
	///</summary>
	vector<pair<string, string>> _vehicleType2vehicleTypeGroup;

	///<summary>
	/// first: ChargingStation
	/// second: VehicleType
	///</summary>
	//vector<pair<std::shared_ptr<Stop>, int>> _vehicleType2chargingStation;

#pragma endregion

};
