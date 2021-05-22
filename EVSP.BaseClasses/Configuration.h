#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include "RecentFile.h"
#include "Typedefs.h"
#include "ConfigSettings.h"


class Configuration
{
public:
	Configuration(std::string workingDirectory);
	~Configuration();

	static std::shared_ptr<Configuration> createConfiguration(std::string workingDirectory, bool cudaAvailability);
	static std::shared_ptr<Configuration> getConfiguration();
	static std::string Filename;

	std::string getWorkingDirectory() const { return _workingDirectory; }

	/// <summary>
	/// Liefert die Anzahl der Threads, die auf der Plattform CPU verwendet werden sollen.
	/// Config-File-Parameter: maxNumOfThreads, minNumOfThreads
	/// </summary>
	int getNumOfThreads() const;

	/// <summary>
	/// Definiert, ob die Haltestellennamen durch zufällig ausgewählte Strassennamen ersetzt werden.
	/// </summary>
	bool getUseRandomStreetnames() const { return _useRandomStreetnames; }

	/// <summary>
	/// Liefert den Dateinamen des Datei, in welcher die Strassennamen für die Benennung anonymisierter Haltestellen enthalten sind.
	/// </summary>
	std::string getStreetnameFile() const { return _streetnameFile; }

	/// <summary>
	/// Die Input-Datei enthält möglicherweise nicht für alle Haltestellen, an denen Servicefahrten enden eine Verbindungsfahrt zu einem
	/// in der Input-Datei definierten Depot. Wenn 'CheckServiceTrips' gleich 'true' ist wird dies überprüft und falls nötig werden
	/// neue Verbindungsfahrten erzeugt, die die Haltestellen der betroffenen Servicefahrten mit Depots verbinden.
	/// </summary>
	bool getCheckServiceTrips() const { return _checkServiceTrips; }

	/// <summary>
	/// Die Input-Datei enthält möglicherweise Verbindungsfahrten, deren Dauer oder Laufzeit mit 0 angegeben ist. Wenn 'CheckEmptyTrips'
	/// gleich 'true' ist wird dies überprüft und falls nötig werden die Dauer bzw. die Laufzeit mit Durchschnittswerden überschrieben.
	/// </summary>
	bool getCheckEmptyTrips() const { return _checkEmptyTrips; }

	/// <summary>
	/// Liefert die Liste der zuletzt verwendeten Files, welche im Config-File gespeichert ist.
	/// Es können maximal 9 Files verwaltet werden.
	/// </summary>
	std::vector<std::shared_ptr<RecentFile>> getRecentFiles() const { return _recentFiles; }

	/// <summary>
	/// Definiert das Lösungsverfahren.
	/// </summary>
	SolverStrategyConfig getSolver() const { return _solver; }

	/// <summary>
	/// Definiert, nach welchem Verfahren die Depots festgelegt werden.
	/// </summary>
	DepotDetectionMode getDepotDetectionMode() const { return _depotDetectionMode; }

	/// <summary>
	/// Definiert, nach welchem Verfahren die Ladestationen festgelegt werden.
	/// </summary>
	ChargingStationDetectionMode getChargingStationDetectionMode() const { return _chargingStationDetectionMode; }

	/// <summary>
	/// 
	/// </summary>
	PlattformConfig getPlattform() const { return _plattform; }

	/// <summary>
	/// Prozentualer Anteil von Ladestationen an der Gesamtanzahl der Bushaltestellen im Bereich 0-100.
	/// Nur relevant, wenn 'charging_station_detection_mode' gleich 'random' ist.
	/// 0: Über die Depots hinaus gibt es keine Ladestationen.
	/// 100: Jede Bushaltestelle ist auch eine Ladestation.
	/// Default: 10%
	/// </summary>
	int getChargingStationRatio() const { return _chargingStationRatio; }

	bool getTerminateOnKeystroke() const { return _terminateOnKeystroke; }

	float getTerminateAfterTime() const { return _terminateAfterTime; }

	int getTerminateAfterRounds() const { return _terminateAfterRounds; }

	bool getPerformChecks() const { return _performChecks; }

	bool isVerboseModeActive()  const { return _verbose; }

	bool getStopwatch() const { return _stopwatch; }

	SaParams getSaParams() const { return _saParams; }

	AcoParams getAcoParams() const { return _acoParams; }

	/// <summary>
	/// Liefert die für die Neuberechnung von Pheromonspuren relevanten Faktoren.
	/// </summary>
	AcoQualifiers getQualifiers() const;

	/// <summary>
	/// 
	/// </summary>
	bool setKeyValue(std::string key, std::string value);

	/// <summary>
	/// 
	/// </summary>
	bool setKeyValue(std::map<std::string, std::string> &keyValues);

	void printConfig(std::ostream &outStream);

private:
	bool load();
	void save();
	static std::shared_ptr<Configuration> _configuration;
	std::string _workingDirectory;
	bool _checkServiceTrips;
	bool _checkEmptyTrips;
	bool _useRandomStreetnames;
	std::string _streetnameFile;
	std::vector<std::shared_ptr<RecentFile>> _recentFiles;
	DepotDetectionMode _depotDetectionMode;
	ChargingStationDetectionMode _chargingStationDetectionMode;
	SolverStrategyConfig _solver;
	PlattformConfig _plattform;
	int _chargingStationRatio;
	bool _terminateOnKeystroke;
	float _terminateAfterTime;
	int _terminateAfterRounds;
	bool _performChecks;
	AcoQualifiers _acoQualifiers;
	SaParams _saParams;
	AcoParams _acoParams;
	int _maxNumOfThreads;
	int _minNumOfThreads;
	int _threadFactor;
	bool _stopwatch;
	bool _verbose;
};

