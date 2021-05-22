#pragma once

#include <limits.h>
#include <iostream>
#include <fstream>


enum DepotDetectionMode {
	/// <summary>
	/// Jede Haltestelle, an der keine Servicefahrten ankommen oder abgehen wird als Depot angesehen.
	/// </summary>
	NO_SERVICE_TRIP,

	/// <summary>
	/// Jede Haltestelle mit einstelliger Legacy-Id wird als Depot angesehen.
	/// </summary>
	ONE_DIGIT_ID
};

enum ChargingStationDetectionMode {
	///<summary>
	/// Es wird ein VSP f�r konventionelle Fahrzeuge gel�st. Keine Ladestationen und unbegrenzte Batteriekapazit�t.
	///</summary>
	NO_CHARGING_STATIONS,

	/// <summary>
	/// Zuf�llig ausgew�hlte Haltestellen werden zu Ladestationen (prozentualer Anteil: ChargingStationRatio).
	/// </summary>
	RANDOM_STOPS,

	/// <summary>
	/// Jede Haltestelle mit einer VehCapacityForCharging > 0 wird zu Ladestation.
	/// </summary>
	VEH_CAPACITY_AT_STOPS,

	/// <summary>
	/// Jede Haltestelle mit einem Eintrag im Block VEHTYPETOCHARGINGSTATION wird zu Ladestation.
	/// </summary>
	VEH_TYPE_ENTRY
};

enum SolverStrategyConfig {
	/// <summary>
	/// L�sungsstrategie unbekannt oder noch nicht ausgew�hlt.
	/// </summary>
	NO_SOLVER,

	/// <summary>
	/// Ant Colony Optimization
	/// </summary>
	ACO,

	/// <summary>
	/// Simulated Annealing
	/// </summary>
	SA
};

enum PlattformConfig {

	/// <summary>
	/// F�r die Komponente ist nicht spezifiziert, wo sie ausgef�hrt wird.
	/// </summary>
	UNDEFINED,

	/// <summary>
	/// Die Komponente wird (soweit m�glich) auf der CPU ausgef�hrt.
	/// </summary>
	CPU,

	/// <summary>
	/// Die Komponente wird (soweit m�glich) auf der GPU ausgef�hrt.
	/// </summary>
	GPU
};


/// <summary>
/// Konfigurationsparameter f�r das Optimierungsverfahren "Ant Colony Optimization".
/// </summary>
struct AcoParams {
	AcoParams();
	void print(std::ostream &outStream);

	/// <summary>
	/// Enth�lt die Populationsgr��e. Also die Anzahl der L�sungsalternativen, die innerhalb eines Durchlaufs 
	/// gleichzeitig untersucht werden.
	/// </summary>
	int populationSize;

	bool printStats;
	bool performChecks;
	bool dumpDecisionNet;
	bool dumpBestSolution;
	bool dumpWorstSolution;

	/// <summary>
	/// Die gr�sste erwartete Anzahl von Uml�ufen in einer L�sung. Zu kleine Werte f�hren dazu, dass das Programm nicht korrekt arbeitet.
	/// Zu grosse Werte f�hren zu einem unn�tig gro�en Speicherverbrauch.
	/// Default: 700
	/// </summary>
	int maxNumOfCirculations;

	/// <summary>
	/// Maximale L�nge eines Umlaufs. Beeinflusst den Speicherbedarf des Programms. Zu kleine Werte
	/// f�hren dazu, dass das Programm nicht korrekt arbeitet.
	/// Default: 50
	/// </summary>
	int maxCirculationLength;

	/// <summary>
	/// Prozentwert zwischen 0.0 (0%) und 1.0 (100%), der angibt, ab welchen Batterieladestand das Aufsuchen von Ladestationen erlaubt ist.
	/// </summary>
	float chargeLevel;

	/// <summary>
	/// Steuert, ob die Kantengewichte des Entscheidungsnetzes beim Auffinden einer neuen Best-L�sung auf das Intervall [0;1] normalisiert 
	/// werden oder nicht.
	/// </summary>
	bool normalizeEdgeWeights;

	bool keepBestSolution;
};


/// <summary>
/// Konfigurationsparameter f�r das Optimierungsverfahren "Simulated Annealing".
/// </summary>
struct SaParams {
	SaParams();
	void print(std::ostream &outStream);

	/// <summary>
	/// Enth�lt die Populationsgr��e. Also die Anzahl der L�sungsalternativen, die innerhalb eines Durchlaufs 
	/// gleichzeitig untersucht werden.
	/// </summary>
	int populationSize;

	/// <summary>
	/// Maximale L�nge eines Umlaufs. Beeinflusst den Speicherbedarf des Programms. Zu kleine Werte
	/// f�hren dazu, dass das Programm nicht korrekt arbeitet.
	/// Default: 50
	/// </summary>
	int maxCirculationLength;

	float minTemperature;
	float startTemperature;
	float crossoverRate;
	int crossoverUpperBound;
	float coolingRate;
	bool greedyCreation;
	float insertionRate;
	int insertionUpperBound;
	float deletionRate;
	int deletionsLowerBound;
	int deletionsUpperBound;
	float greedyInsertion;
	float greedyInsertionAfterDeletion;
	float crossoverChance;
	float insertionChance;
	float circCostDeletionChance;
	float numOfServiceTripsDeletionChance;
	float randomDeletionChance;

	bool printStats;
	bool performChecks;
};


struct AcoQualifiers {
	AcoQualifiers();
	void print(std::ostream &outStream) const;

	/// <summary>
	/// Initialwert f�r das Kantengewicht. Wird mit der Gr��e der Population multipliziert, damit die Relationen auch bei 
	/// gro�en Populationen erhalten bleiben.
	/// Default: 1,0
	/// </summary>
	float initialWeight;

	/// <summary>
	/// Prozentsatz, um dem die bereits bestehenden Markierungen verringert werden, bevor eine neue Spur angebracht wird.
	/// Der Wert 1 (100%) l�scht die Spuren der vorausgehenden Runde.
	/// Der Wert 0.05 (5%) verringert die Spuren der vorausgehenden Runden um 5%.
	/// Der Wert 0 (0%) l�sst alte Spuren unbegrenzt lange weiter bestehen.
	/// Default: 0.1
	/// </summary>
	float fading;

	/// <summary>
	/// Faktor innerhalb der Formel f�r die Berechnung der neuen Pheromonspur, der bestimmt, wie stark die Gesamtkosten
	/// der jeweiligen L�sung gewichtet werden.
	/// Default: 0.1
	/// </summary>
	float totalCostQualifier;

	/// <summary>
	/// Faktor innerhalb der Formel f�r die Berechnung der neuen Pheromonspur, der bestimmt, wie stark die Fahrzeuganzahl
	/// der jeweiligen L�sung gewichtet wird.
	/// Default: 0.1
	/// </summary>
	float numOfVehiclesQualifier;

	/// <summary>
	/// Faktor innerhalb der Formel f�r die Berechnung der neuen Pheromonspur, der bestimmt, wie stark das 
	/// Brutto-Netto-Kostenverh�ltnis des jeweiligen Umlaufs gewichtet wird.
	/// Default: 1.0
	/// </summary>
	float circCostRatioQualifier;

	/// <summary>
	/// Faktor innerhalb der Formel f�r die Berechnung der neuen Pheromonspur, der die Verst�rkung der Spur einer gerade neu 
	/// gefundenen besten L�sung bestimmt.
	/// gewichtet wird.
	/// 1.0: Keine Verst�rkung (100%). 
	/// 1.5: Verst�rkung um 50% (auf 150%). 
	/// 2.0: Verst�rkung um 100% (auf das Doppelte).
	/// Default: 1.0
	/// </summary>
	//float strengthenTheBestSolution;

	/// <summary>
	/// Faktor innerhalb der Formel f�r die Berechnung der neuen Pheromonspur, der die Abschw�chung der Spur einer in einer 
	/// vorangegangenen Runde gefundenen besten L�sung bestimmt.
	/// 1.0: Keine Abschw�chung. 
	/// 0.5: Abschw�chung um 50%. 
	/// 0.0: Abschw�chung um 100% (auf 0). 
	/// Default: 1.0
	/// </summary>
	//float weakenTheBestSolution;

	/// <summary>
	/// Bestimmt, wie stark die Spuren "schlechter" L�sungen im Verh�ltnis zu den Spuren "guter" L�sungen abgeschw�cht werden.
	/// Dies erfolgt durch die Potenzierung des Teilergebnisses mit dem hier gelieferten Wert. Da das Teilergebnis auf Werte
	/// zwischen 0 und 1 normiert wurde f�hrt ein hoher Wert zu einer starken Abschw�chung "schlechter" Spuren.
	/// Beim Wert 1 erfolgt keine Abschw�chung. Werte unter 1 f�hren zu einer �berbetonung guter Werte. Werte �ber 1 f�hren 
	/// zu einer �berbetonung schlechter Werte.
	/// Default: 1.0
	/// </summary>
	float weakenAllBadSolutions;
};
