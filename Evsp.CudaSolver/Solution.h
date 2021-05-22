#pragma once

#include <list>
#include "RandomCpu.h"
#include "CuServiceTrips.h"
#include "CuProblem.h"
#include "Circulation.h"
#include "MatrixCreator.h"
#include "DepotTools.h"
#include "SolutionStats.h"

using namespace std;

/// <summary>
/// Enthält sämtliche Umläufe einer Lösung. 
/// </summary>
class Solution
{
public:

	//Solution();
	Solution() = delete;
	Solution(const Solution &other);
	Solution(int maxNumOfActions, int numOfServiceTrips);
	~Solution();

	Solution& operator=(const Solution &rhs);

	bool operator==(Solution &rhs);
	bool operator!=(Solution &rhs);

	bool check(shared_ptr<CuProblem> problem);

	void dump();

	SolutionStats getStats(shared_ptr<CuProblem> problem);

	/// <summary>
	/// Liefert die Anzahl der Umläufe einer Lösung.
	/// </summary>
	/// <returns>Anzahl der Umläufe</returns>
	int getNumOfCirculations() const;

	/// <summary>
	/// Liefert die Anzahl der Aktionen eines Umlaufs.
	/// </summary>
	/// <param name="circulationId">Id des Umlaufs</param>
	/// <returns>Anzahl der Aktionen</returns>
	int getNumOfActions(CirculationId circulationId) const;

	/// <summary>
	/// Liefert die Haltestellen-ID des Depots, das einem Umlauf zugeordnet ist.
	/// </summary>
	/// <param name="circulationId">Id des Umlaufs</param>
	/// <returns></returns>
	StopId getDepotId(CirculationId circulationId) const;

	/// <summary>
	/// Liefert die ID des Fahrzeugtyps, der einem Umlauf zugeordnet ist.
	/// </summary>
	/// <param name="circulationId">Id des Umlaufs</param>
	/// <returns>Fahrzeugtyp-ID des Umlaufs.</returns>
	VehicleTypeId getVehicleTypeId(CirculationId circulationId) const;

	/// <summary>
	/// Fügt einen neuen (leeren) Umlauf hinzu.
	/// </summary>
	/// <param name="depotId">Id des Depots, in dem der neue Umlauf startet und endet.</param>
	/// <param name="vehicleTypeId">Id des Fahrzeugtyps, mit dem der neue Umlauf absolviert wird.</param>
	CirculationId createNewCirculation(StopId depotId, VehicleTypeId vehicleTypeId);

	/// <summary>
	/// Fügt einen neuen Umlauf hinzu, der genau eine (nämlich die übergebene) Servicefahrt enthält.
	/// Depot und Fahrzeugtyp werden automatisch gewählt (Greedy-Algorithmus).
	/// </summary>
	/// <returns>Liefert die Id des gerade erstellen Umlaufs zurück.</returns>
	CirculationId createNewCirculation(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem);

	/// <summary>
	/// Fügt einen neuen Umlauf hinzu.
	/// </summary>
	CirculationId addCirculation(Circulation &newCirculation);

	/// <summary>
	/// Liefert den Umlauf mit der übergebenen Id.
	/// </summary>
	Circulation& getCirculation(CirculationId circulationId);
	const Circulation& getCirculation(CirculationId circulationId) const;

	/// <summary>
	/// Fügt einer Lösung eine bislang noch nicht eingeplante Servicefahrt hinzu. Dabei wird zunächst versucht, die SF in einen bestehenden
	/// Umlauf einzufügen. Wenn das nicht möglich ist wird ein neuer Umlauf erzeugt, der die SF aufnimmt. Würde die SF in mehrere bestehende
	/// Umläufe passen, wird der Umlauf ausgewählt, bei dem das Hinzufügen der SF die geringsten Kosten verursacht.
	/// </summary>
	void insertServiceTripGreedy(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem);

	/// <summary>
	/// Fügt einer Lösung eine bislang noch nicht eingeplante Servicefahrt hinzu. Dabei wird zunächst versucht, die SF in einen 
	/// bestehenden Umlauf einzufügen. Wenn das nicht möglich ist wird ein neuer Umlauf erzeugt, der die SF aufnimmt. Würde die 
	/// SF in mehrere bestehende Umläufe passen erfolgt eine zufällige Auswahl.
	/// </summary>
	void insertServiceTripRandom(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem, RandomCpu &randCpu);

	/// <summary>
	/// Teilt zwei Umläufe in jeweils zwei Teile ("Kopf" und "Schwanz") und fügt den Kopf des einen mit dem Schwanz des anderen Umlaufs
	/// zu zwei neuen Umläufen zusammen.
	/// </summary>
	void performCrossoverOperation(int numOfServiceTrips, shared_ptr<CuProblem> problem, float crossoverRate, int crossoverUpperBound, RandomCpu &randCpu);

	/// <summary>
	/// Entnimmt eine Menge zufällig ausgewählter Servicefahrten aus den Umläufen, in denen sie bislang enthalten sind und fügt sie
	/// dann in andere Umläufe ein.
	/// </summary>
	void performInsertionOperation(bool greedyInsertion, int numOfServiceTrips, float insertionRate, int insertionUpperBound, shared_ptr<CuProblem> problem, RandomCpu &randCpu);

	/// <summary>
	/// Definiert, nach welchem Kriterium der Deletion-Operator entscheidet, ob ein Umlauf gelöscht werden soll oder nicht.
	/// </summary>
	enum DeletionMode {
		RANDOM,					// Rein zufällige Auswahl der zu löschenden Umläufe. Die Wahrscheinlichkeit ist für alle Umläufe gleich hoch.
		CIRC_COST_RATIO,		// Kostenverhältnis: Umläufe mit schlechtem Kostenverhältnis werden mit höherer Wahrscheinlichkeit gelöscht.
		NUM_OF_SERVICE_TRIPS	// Anzahl der Servicefahrten: "Kurze" Umläufe werden mit höherer Wahrscheinlichkeit gelöscht.
	};

	/// <summary>
	/// Löscht eine Menge zufällig ausgewählter Umläufe. Bei der Auswahl der Umläufe werden ggf. Parameter wir das Kostenverhältnis
	/// eines Umlaufs berücksichtigt. Die vormals in den gelöschten Umläufen enthaltenen Servicefahrten werden auf die verbliebenen 
	/// Umläufe verteilt. Nötigenfalls werden neue Umläufe erzeugt, um die Servicefahrten aufzunehmen.
	/// </summary>
	/// <param name="deletionRate">Prozentualer Anteil der Menge der zu löschenden Umläufe an der Gesamtmenge der Umläufe (0.0 bis 1.0).</param>
	/// <param name="lowerBound">Die unabhängig von 'deletionRate' mindestens zu löschende Anzahl von Umläufen.</param>
	/// <param name="upperBound">Die unabhängig von 'deletionRate' höchstens zu löschende Anzahl von Umläufen.</param>
	/// <param name="deletionMode">Legt fest, nach welchem Kriterium ein Umlauf bewertet wird.</param>
	/// <param name="greedyInsertion">Wenn TRUE werden die frei gewordenen Servicefahrten so auf Umläufe verteilt, dass die geringst-
	/// möglichen Gesamtkosten entstehen. Andernfalls erfolgt die Verteilung rein zufällig.</param>
	void performDeleteOperation(float deletionRate, int deletionsLowerBound, int upperBound, DeletionMode deletionMode,
		bool greedyInsertion, shared_ptr<CuProblem> problem, RandomCpu &randCpu);

	/// <summary>
	/// Löscht einen Umlauf. Dadurch wird die Lösung ungültig, da nicht mehr alle Servicefahrten verplant sind.
	/// Die im Umlauf enthaltenen Servicefahrten werden in einem Vektor zurückgeliefert und müssen
	/// im Anschluss wieder in Umläufe eingegliedert werden um die Lösung wieder gültig zu machen.
	/// ACHTUNG: Die Id des zu löschenden Umlaufs wird während der Operation entweder ungültig oder
	/// einem anderen Umlauf zugeordnet. Auch andere Ids auf Umläufe können während der Operation ungültig werden, weshalb 
	/// vorher zugewiesene Ids nach dem Aufruf nicht mehr verwendet werden dürfen!
	/// </summary>
	/// <param name="circToBeDeleted">Id des zu löschenden Umlaufs. Die Id wird während der Operation entweder ungültig oder
	/// einem anderen Umlauf zugeordnet. Auch andere Ids auf Umläufe können während der Operation ungültig werden, weshalb 
	/// vorher zugewiesene Ids nach dem Aufruf nicht mehr verwendet werden dürfen!</param>
	/// <param name="leftFreeServiceTrips">Vektor, der die Ids der im Umlauf enthaltenen Servicefahrten aufnimmt.</param>
	/// <returns>Liefert 'leftFreeServiceTrips' zurück.</returns>
	vector<ServiceTripId>& removeCirculation(CirculationId circToBeDeleted, vector<ServiceTripId> &leftFreeServiceTrips);

	/// <summary>
	/// Prüft, ob der Umlauf Servicefahrten enthält. Löscht den Umlauf, wenn das nicht der Fall ist.
	/// ACHTUNG: Die Id des zu löschenden Umlaufs wird während der Operation ggf. ungültig oder
	/// einem anderen Umlauf zugeordnet. Auch andere Ids auf Umläufe können während der Operation ungültig werden, weshalb 
	/// vorher zugewiesene Ids nach dem Aufruf nicht mehr verwendet werden dürfen!
	/// </summary>
	/// <returns>Liefert TRUE, falls der Umlauf gelöscht wurde.</returns>
	bool removeCirculationIfEmpty(CirculationId id);

private:
	//void checkId();

	/// <summary>
	/// Liefert einen Vektor, der die Zuordnung der Servicefahrten zu den Umläufen enthält.
	/// </summary>
	vector<CirculationId> getServiceTripToCirculationVector(shared_ptr<CuProblem> problem);
		
	int _maxNumOfActions;

	// Daten sämtlicher Umläufe.
	list<pair<CirculationId, Circulation>> _circulations;

	// Circulation-Id für den nächsten neuen Umlauf.
	CirculationId _nextCirculationId;
};
