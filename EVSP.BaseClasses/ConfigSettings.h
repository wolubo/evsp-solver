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
	/// Es wird ein VSP für konventionelle Fahrzeuge gelöst. Keine Ladestationen und unbegrenzte Batteriekapazität.
	///</summary>
	NO_CHARGING_STATIONS,

	/// <summary>
	/// Zufällig ausgewählte Haltestellen werden zu Ladestationen (prozentualer Anteil: ChargingStationRatio).
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
	/// Lösungsstrategie unbekannt oder noch nicht ausgewählt.
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
	/// Für die Komponente ist nicht spezifiziert, wo sie ausgeführt wird.
	/// </summary>
	UNDEFINED,

	/// <summary>
	/// Die Komponente wird (soweit möglich) auf der CPU ausgeführt.
	/// </summary>
	CPU,

	/// <summary>
	/// Die Komponente wird (soweit möglich) auf der GPU ausgeführt.
	/// </summary>
	GPU
};


/// <summary>
/// Konfigurationsparameter für das Optimierungsverfahren "Ant Colony Optimization".
/// </summary>
struct AcoParams {
	AcoParams();
	void print(std::ostream &outStream);

	/// <summary>
	/// Enthält die Populationsgröße. Also die Anzahl der Lösungsalternativen, die innerhalb eines Durchlaufs 
	/// gleichzeitig untersucht werden.
	/// </summary>
	int populationSize;

	bool printStats;
	bool performChecks;
	bool dumpDecisionNet;
	bool dumpBestSolution;
	bool dumpWorstSolution;

	/// <summary>
	/// Die grösste erwartete Anzahl von Umläufen in einer Lösung. Zu kleine Werte führen dazu, dass das Programm nicht korrekt arbeitet.
	/// Zu grosse Werte führen zu einem unnötig großen Speicherverbrauch.
	/// Default: 700
	/// </summary>
	int maxNumOfCirculations;

	/// <summary>
	/// Maximale Länge eines Umlaufs. Beeinflusst den Speicherbedarf des Programms. Zu kleine Werte
	/// führen dazu, dass das Programm nicht korrekt arbeitet.
	/// Default: 50
	/// </summary>
	int maxCirculationLength;

	/// <summary>
	/// Prozentwert zwischen 0.0 (0%) und 1.0 (100%), der angibt, ab welchen Batterieladestand das Aufsuchen von Ladestationen erlaubt ist.
	/// </summary>
	float chargeLevel;

	/// <summary>
	/// Steuert, ob die Kantengewichte des Entscheidungsnetzes beim Auffinden einer neuen Best-Lösung auf das Intervall [0;1] normalisiert 
	/// werden oder nicht.
	/// </summary>
	bool normalizeEdgeWeights;

	bool keepBestSolution;
};


/// <summary>
/// Konfigurationsparameter für das Optimierungsverfahren "Simulated Annealing".
/// </summary>
struct SaParams {
	SaParams();
	void print(std::ostream &outStream);

	/// <summary>
	/// Enthält die Populationsgröße. Also die Anzahl der Lösungsalternativen, die innerhalb eines Durchlaufs 
	/// gleichzeitig untersucht werden.
	/// </summary>
	int populationSize;

	/// <summary>
	/// Maximale Länge eines Umlaufs. Beeinflusst den Speicherbedarf des Programms. Zu kleine Werte
	/// führen dazu, dass das Programm nicht korrekt arbeitet.
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
	/// Initialwert für das Kantengewicht. Wird mit der Größe der Population multipliziert, damit die Relationen auch bei 
	/// großen Populationen erhalten bleiben.
	/// Default: 1,0
	/// </summary>
	float initialWeight;

	/// <summary>
	/// Prozentsatz, um dem die bereits bestehenden Markierungen verringert werden, bevor eine neue Spur angebracht wird.
	/// Der Wert 1 (100%) löscht die Spuren der vorausgehenden Runde.
	/// Der Wert 0.05 (5%) verringert die Spuren der vorausgehenden Runden um 5%.
	/// Der Wert 0 (0%) lässt alte Spuren unbegrenzt lange weiter bestehen.
	/// Default: 0.1
	/// </summary>
	float fading;

	/// <summary>
	/// Faktor innerhalb der Formel für die Berechnung der neuen Pheromonspur, der bestimmt, wie stark die Gesamtkosten
	/// der jeweiligen Lösung gewichtet werden.
	/// Default: 0.1
	/// </summary>
	float totalCostQualifier;

	/// <summary>
	/// Faktor innerhalb der Formel für die Berechnung der neuen Pheromonspur, der bestimmt, wie stark die Fahrzeuganzahl
	/// der jeweiligen Lösung gewichtet wird.
	/// Default: 0.1
	/// </summary>
	float numOfVehiclesQualifier;

	/// <summary>
	/// Faktor innerhalb der Formel für die Berechnung der neuen Pheromonspur, der bestimmt, wie stark das 
	/// Brutto-Netto-Kostenverhältnis des jeweiligen Umlaufs gewichtet wird.
	/// Default: 1.0
	/// </summary>
	float circCostRatioQualifier;

	/// <summary>
	/// Faktor innerhalb der Formel für die Berechnung der neuen Pheromonspur, der die Verstärkung der Spur einer gerade neu 
	/// gefundenen besten Lösung bestimmt.
	/// gewichtet wird.
	/// 1.0: Keine Verstärkung (100%). 
	/// 1.5: Verstärkung um 50% (auf 150%). 
	/// 2.0: Verstärkung um 100% (auf das Doppelte).
	/// Default: 1.0
	/// </summary>
	//float strengthenTheBestSolution;

	/// <summary>
	/// Faktor innerhalb der Formel für die Berechnung der neuen Pheromonspur, der die Abschwächung der Spur einer in einer 
	/// vorangegangenen Runde gefundenen besten Lösung bestimmt.
	/// 1.0: Keine Abschwächung. 
	/// 0.5: Abschwächung um 50%. 
	/// 0.0: Abschwächung um 100% (auf 0). 
	/// Default: 1.0
	/// </summary>
	//float weakenTheBestSolution;

	/// <summary>
	/// Bestimmt, wie stark die Spuren "schlechter" Lösungen im Verhältnis zu den Spuren "guter" Lösungen abgeschwächt werden.
	/// Dies erfolgt durch die Potenzierung des Teilergebnisses mit dem hier gelieferten Wert. Da das Teilergebnis auf Werte
	/// zwischen 0 und 1 normiert wurde führt ein hoher Wert zu einer starken Abschwächung "schlechter" Spuren.
	/// Beim Wert 1 erfolgt keine Abschwächung. Werte unter 1 führen zu einer Überbetonung guter Werte. Werte über 1 führen 
	/// zu einer Überbetonung schlechter Werte.
	/// Default: 1.0
	/// </summary>
	float weakenAllBadSolutions;
};
