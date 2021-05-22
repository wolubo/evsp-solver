#pragma once

#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/PeriodOfTime.hpp"

/// <summary>
/// Statistische Daten zu einem Umlauf: Kosten, Verbrauch, ...
/// </summary>
class CirculationStats
{
public:
	CirculationStats();
	~CirculationStats();

	float getCircCostRatio() const { return (float)totalCost / (float)serviceTripCosts; }

	/// <summary>
	/// Zeitraum, in dem der Umlauf stattfindet.
	/// </summary>
	PeriodOfTime time;

	/// <summary>
	/// Die während des Umlaufs insgesamt zurückgelegte Strecke in Metern.
	/// </summary>
	DistanceInMeters totalDistance;

	/// <summary>
	/// Die Gesamtdauer des Umlaufs in Sekunden.
	/// </summary>
	DurationInSeconds totalDuration;

	/// <summary>
	/// Die Gesamtdauer aller Servicefahrten des Umlaufs in Sekunden.
	/// </summary>	
	DurationInSeconds serviceTripDuration;

	/// <summary>
	/// Enthält die Gesamtkosten des (Teil-)Umlaufs. Das ist die Summe aus den Kosten für Aufladungen sowie den zeit- und streckenabhängigen
	/// Kosten. Die Anschaffungskosten des verwendeten Fahrzeugs gehören bei Teilumläufen (Circulation) nicht dazu. Bei 
	/// vollständigen Umläufen (Circulation) werden die Anschaffungskosten jedoch hinzugezählt.
	/// </summary>
	AmountOfMoney totalCost;

	/// <summary>
	/// Enthält die zeitabhängigen Kosten des (Teil-)Umlaufs.
	/// </summary>
	AmountOfMoney timeDependentCosts;

	/// <summary>
	/// Enthält die streckenabhängigen Kosten des (Teil-)Umlaufs. Dazu gehören auch die Kosten aller Aufladungen.
	/// </summary>
	AmountOfMoney distanceDependentCosts;

	/// <summary>
	/// Die Gesamtkosten aller Servicefahrten des Umlaufs in Sekunden.
	/// </summary>	
	AmountOfMoney serviceTripCosts;

	/// <summary>
	/// Enthält die Batteriekapazität, die vor Beginn des (Teil-)Umlaufs mindestens vorhanden sein muss, damit der Umlauf erfolgreich 
	/// durchgeführt werden kann. Also den Verbrauch aller Aktionen bis zum Erreichen einer Ladestation bzw. bis zur Rückkehr ins Depot.
	/// </summary>
	KilowattHour minimumCapacity;

	/// <summary>
	/// Enthält die nach Abschluss des (Teil-)Umlaufs noch verfügbare Batteriekapazität.
	/// </summary>
	KilowattHour remainingCapacity;
};
