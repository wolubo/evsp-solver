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
	/// Die w�hrend des Umlaufs insgesamt zur�ckgelegte Strecke in Metern.
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
	/// Enth�lt die Gesamtkosten des (Teil-)Umlaufs. Das ist die Summe aus den Kosten f�r Aufladungen sowie den zeit- und streckenabh�ngigen
	/// Kosten. Die Anschaffungskosten des verwendeten Fahrzeugs geh�ren bei Teiluml�ufen (Circulation) nicht dazu. Bei 
	/// vollst�ndigen Uml�ufen (Circulation) werden die Anschaffungskosten jedoch hinzugez�hlt.
	/// </summary>
	AmountOfMoney totalCost;

	/// <summary>
	/// Enth�lt die zeitabh�ngigen Kosten des (Teil-)Umlaufs.
	/// </summary>
	AmountOfMoney timeDependentCosts;

	/// <summary>
	/// Enth�lt die streckenabh�ngigen Kosten des (Teil-)Umlaufs. Dazu geh�ren auch die Kosten aller Aufladungen.
	/// </summary>
	AmountOfMoney distanceDependentCosts;

	/// <summary>
	/// Die Gesamtkosten aller Servicefahrten des Umlaufs in Sekunden.
	/// </summary>	
	AmountOfMoney serviceTripCosts;

	/// <summary>
	/// Enth�lt die Batteriekapazit�t, die vor Beginn des (Teil-)Umlaufs mindestens vorhanden sein muss, damit der Umlauf erfolgreich 
	/// durchgef�hrt werden kann. Also den Verbrauch aller Aktionen bis zum Erreichen einer Ladestation bzw. bis zur R�ckkehr ins Depot.
	/// </summary>
	KilowattHour minimumCapacity;

	/// <summary>
	/// Enth�lt die nach Abschluss des (Teil-)Umlaufs noch verf�gbare Batteriekapazit�t.
	/// </summary>
	KilowattHour remainingCapacity;
};
