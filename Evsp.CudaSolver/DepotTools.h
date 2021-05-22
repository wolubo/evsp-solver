#pragma once

#include <memory>
#include <assert.h>
#include "EVSP.BaseClasses/DeviceLaunchParameters.h"
#include "EVSP.BaseClasses/Typedefs.h"
#include "CuProblem.h"
#include "CuAction.h"
#include "Circulation.h"

class DepotTools
{
public:
	DepotTools() = delete;
	DepotTools(DepotTools&) = delete;
	DepotTools(shared_ptr<CuProblem> problem);
	~DepotTools();

	/// <summary>
	/// Prüft, ob sich der Umlauf bzw. der Teilumlauf von einem bestimmten Depot aus durchführen lässt.
	/// </summary>			
	bool proofDepot(Circulation &circ, StopId depotId);

	/// <summary>
	/// Prüft, ob dem Umlauf bereits ein Depot zugeordnet ist und ob die Fahrt im Depot beginnt und endet. 
	/// Versucht im Fehlerfall, ein passendes Depot auszuwählen und Aus- und Einrückfahrten hinzuzufügen.
	/// Es kann nach der Reparatur nötig sein, das Fahrzeug aufzuladen, weil die Restkapazität nicht mehr
	/// für die Fahrt zum Depot ausreicht. Daher sollte im Anschluss ein Aufruf der Methode repairChargingStationVisits()
	/// erfolgen!
	/// </summary>			
	/// <param name="circ">Der zu reparierende Umlauf. Falls nötig wird das Depot geändertund es werden eine 
	/// Aus- und/oder eine Einrückfahrt hinzugefügt.</param>
	/// <returns>TRUE, wenn keine Fehler (mehr) vorliegen.</returns>
	bool repairDepot(Circulation &circ);

	/// <summary>
	/// Findet das Depot, von dem aus sich ein Umlauf mit einem bestimmten Fahrzeugtypen zu den geringsten Kosten durchführen lässt.
	/// </summary>
	/// <param name="start">Haltestellen-Id der Starthaltestelle des Umlaufs</param>
	/// <param name="end">Haltestellen-Id der Endhaltestelle des Umlaufs</param>
	/// <param name="vehicleTypeId">Id des Fahrzeugtypen, mit dem der Umlauf durchgeführt wird</param>
	/// <returns>Haltestellen-Id des Depots. Ungültige Id, falls kein Depot gefunden werden kann.</returns>
	CU_HSTDEV StopId findBestDepot(StopId start, StopId end, VehicleTypeId vehicleTypeId) const;

	/// <summary>
	/// Findet das Depot, von dem aus sich der übergebene Umlauf mit einem bestimmten Fahrzeugtypen 
	/// zu den geringsten Kosten durchführen lässt.
	/// </summary>
	/// <param name="circ">Umlauf, zu dem ein neues Depot gefunden werden soll.</param>
	/// <returns>Haltestellen-Id des Depots. Ungültige Id, falls kein Depot gefunden werden kann.</returns>
	CU_HSTDEV StopId findBestDepot(Circulation &circ) const;

	/// <summary>
	/// Findet ein Depot, von dem aus sich ein Umlauf durchführen lässt. Falls mehrere Depots in frage kommen erfolgt eine Zufallsauswahl.
	/// </summary>
	/// <param name="start">Haltestellen-Id der Starthaltestelle des Umlaufs</param>
	/// <param name="end">Haltestellen-Id der Endhaltestelle des Umlaufs</param>
	/// <param name="exclude">Haltestellen-Id eines Depots, dass nicht gewählt (also ausgeschlossen) werden soll.</param>
	/// <param name="randomNumber">Zufallszahl zwischen 0 und MAX_INT, die vorab bestimmt werden muss</param>
	/// <returns>Haltestellen-Id des Depots. Ungültige Id, falls kein Depot gefunden werden kann.</returns>
	CU_HSTDEV StopId findRandomDepot(StopId start, StopId end, StopId exclude, int randomNumber) const;

	/// <summary>
	/// Findet ein Depot, von dem aus sich der übergebene Umlauf durchführen lässt.
	/// Falls mehrere Depots in frage kommen erfolgt eine Zufallsauswahl. Das aktuelle Depot des Umlaufs wird 
	/// von der Suche ausschlossen.
	/// </summary>
	/// <param name="circ">Umlauf, zu dem ein neues Depot gefunden werden soll.</param>
	/// <param name="randomNumber">Zufallszahl zwischen 0 und MAX_INT, die vorab bestimmt werden muss</param>
	/// <returns>Haltestellen-Id des Depots. Ungültige Id, falls kein Depot gefunden werden kann.</returns>
	CU_HSTDEV StopId findRandomDepot(Circulation &circ, int randomNumber) const;

private:
	shared_ptr<CuProblem> _problem;
};

