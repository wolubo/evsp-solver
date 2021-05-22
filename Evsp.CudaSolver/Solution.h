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
/// Enth�lt s�mtliche Uml�ufe einer L�sung. 
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
	/// Liefert die Anzahl der Uml�ufe einer L�sung.
	/// </summary>
	/// <returns>Anzahl der Uml�ufe</returns>
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
	/// F�gt einen neuen (leeren) Umlauf hinzu.
	/// </summary>
	/// <param name="depotId">Id des Depots, in dem der neue Umlauf startet und endet.</param>
	/// <param name="vehicleTypeId">Id des Fahrzeugtyps, mit dem der neue Umlauf absolviert wird.</param>
	CirculationId createNewCirculation(StopId depotId, VehicleTypeId vehicleTypeId);

	/// <summary>
	/// F�gt einen neuen Umlauf hinzu, der genau eine (n�mlich die �bergebene) Servicefahrt enth�lt.
	/// Depot und Fahrzeugtyp werden automatisch gew�hlt (Greedy-Algorithmus).
	/// </summary>
	/// <returns>Liefert die Id des gerade erstellen Umlaufs zur�ck.</returns>
	CirculationId createNewCirculation(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem);

	/// <summary>
	/// F�gt einen neuen Umlauf hinzu.
	/// </summary>
	CirculationId addCirculation(Circulation &newCirculation);

	/// <summary>
	/// Liefert den Umlauf mit der �bergebenen Id.
	/// </summary>
	Circulation& getCirculation(CirculationId circulationId);
	const Circulation& getCirculation(CirculationId circulationId) const;

	/// <summary>
	/// F�gt einer L�sung eine bislang noch nicht eingeplante Servicefahrt hinzu. Dabei wird zun�chst versucht, die SF in einen bestehenden
	/// Umlauf einzuf�gen. Wenn das nicht m�glich ist wird ein neuer Umlauf erzeugt, der die SF aufnimmt. W�rde die SF in mehrere bestehende
	/// Uml�ufe passen, wird der Umlauf ausgew�hlt, bei dem das Hinzuf�gen der SF die geringsten Kosten verursacht.
	/// </summary>
	void insertServiceTripGreedy(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem);

	/// <summary>
	/// F�gt einer L�sung eine bislang noch nicht eingeplante Servicefahrt hinzu. Dabei wird zun�chst versucht, die SF in einen 
	/// bestehenden Umlauf einzuf�gen. Wenn das nicht m�glich ist wird ein neuer Umlauf erzeugt, der die SF aufnimmt. W�rde die 
	/// SF in mehrere bestehende Uml�ufe passen erfolgt eine zuf�llige Auswahl.
	/// </summary>
	void insertServiceTripRandom(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem, RandomCpu &randCpu);

	/// <summary>
	/// Teilt zwei Uml�ufe in jeweils zwei Teile ("Kopf" und "Schwanz") und f�gt den Kopf des einen mit dem Schwanz des anderen Umlaufs
	/// zu zwei neuen Uml�ufen zusammen.
	/// </summary>
	void performCrossoverOperation(int numOfServiceTrips, shared_ptr<CuProblem> problem, float crossoverRate, int crossoverUpperBound, RandomCpu &randCpu);

	/// <summary>
	/// Entnimmt eine Menge zuf�llig ausgew�hlter Servicefahrten aus den Uml�ufen, in denen sie bislang enthalten sind und f�gt sie
	/// dann in andere Uml�ufe ein.
	/// </summary>
	void performInsertionOperation(bool greedyInsertion, int numOfServiceTrips, float insertionRate, int insertionUpperBound, shared_ptr<CuProblem> problem, RandomCpu &randCpu);

	/// <summary>
	/// Definiert, nach welchem Kriterium der Deletion-Operator entscheidet, ob ein Umlauf gel�scht werden soll oder nicht.
	/// </summary>
	enum DeletionMode {
		RANDOM,					// Rein zuf�llige Auswahl der zu l�schenden Uml�ufe. Die Wahrscheinlichkeit ist f�r alle Uml�ufe gleich hoch.
		CIRC_COST_RATIO,		// Kostenverh�ltnis: Uml�ufe mit schlechtem Kostenverh�ltnis werden mit h�herer Wahrscheinlichkeit gel�scht.
		NUM_OF_SERVICE_TRIPS	// Anzahl der Servicefahrten: "Kurze" Uml�ufe werden mit h�herer Wahrscheinlichkeit gel�scht.
	};

	/// <summary>
	/// L�scht eine Menge zuf�llig ausgew�hlter Uml�ufe. Bei der Auswahl der Uml�ufe werden ggf. Parameter wir das Kostenverh�ltnis
	/// eines Umlaufs ber�cksichtigt. Die vormals in den gel�schten Uml�ufen enthaltenen Servicefahrten werden auf die verbliebenen 
	/// Uml�ufe verteilt. N�tigenfalls werden neue Uml�ufe erzeugt, um die Servicefahrten aufzunehmen.
	/// </summary>
	/// <param name="deletionRate">Prozentualer Anteil der Menge der zu l�schenden Uml�ufe an der Gesamtmenge der Uml�ufe (0.0 bis 1.0).</param>
	/// <param name="lowerBound">Die unabh�ngig von 'deletionRate' mindestens zu l�schende Anzahl von Uml�ufen.</param>
	/// <param name="upperBound">Die unabh�ngig von 'deletionRate' h�chstens zu l�schende Anzahl von Uml�ufen.</param>
	/// <param name="deletionMode">Legt fest, nach welchem Kriterium ein Umlauf bewertet wird.</param>
	/// <param name="greedyInsertion">Wenn TRUE werden die frei gewordenen Servicefahrten so auf Uml�ufe verteilt, dass die geringst-
	/// m�glichen Gesamtkosten entstehen. Andernfalls erfolgt die Verteilung rein zuf�llig.</param>
	void performDeleteOperation(float deletionRate, int deletionsLowerBound, int upperBound, DeletionMode deletionMode,
		bool greedyInsertion, shared_ptr<CuProblem> problem, RandomCpu &randCpu);

	/// <summary>
	/// L�scht einen Umlauf. Dadurch wird die L�sung ung�ltig, da nicht mehr alle Servicefahrten verplant sind.
	/// Die im Umlauf enthaltenen Servicefahrten werden in einem Vektor zur�ckgeliefert und m�ssen
	/// im Anschluss wieder in Uml�ufe eingegliedert werden um die L�sung wieder g�ltig zu machen.
	/// ACHTUNG: Die Id des zu l�schenden Umlaufs wird w�hrend der Operation entweder ung�ltig oder
	/// einem anderen Umlauf zugeordnet. Auch andere Ids auf Uml�ufe k�nnen w�hrend der Operation ung�ltig werden, weshalb 
	/// vorher zugewiesene Ids nach dem Aufruf nicht mehr verwendet werden d�rfen!
	/// </summary>
	/// <param name="circToBeDeleted">Id des zu l�schenden Umlaufs. Die Id wird w�hrend der Operation entweder ung�ltig oder
	/// einem anderen Umlauf zugeordnet. Auch andere Ids auf Uml�ufe k�nnen w�hrend der Operation ung�ltig werden, weshalb 
	/// vorher zugewiesene Ids nach dem Aufruf nicht mehr verwendet werden d�rfen!</param>
	/// <param name="leftFreeServiceTrips">Vektor, der die Ids der im Umlauf enthaltenen Servicefahrten aufnimmt.</param>
	/// <returns>Liefert 'leftFreeServiceTrips' zur�ck.</returns>
	vector<ServiceTripId>& removeCirculation(CirculationId circToBeDeleted, vector<ServiceTripId> &leftFreeServiceTrips);

	/// <summary>
	/// Pr�ft, ob der Umlauf Servicefahrten enth�lt. L�scht den Umlauf, wenn das nicht der Fall ist.
	/// ACHTUNG: Die Id des zu l�schenden Umlaufs wird w�hrend der Operation ggf. ung�ltig oder
	/// einem anderen Umlauf zugeordnet. Auch andere Ids auf Uml�ufe k�nnen w�hrend der Operation ung�ltig werden, weshalb 
	/// vorher zugewiesene Ids nach dem Aufruf nicht mehr verwendet werden d�rfen!
	/// </summary>
	/// <returns>Liefert TRUE, falls der Umlauf gel�scht wurde.</returns>
	bool removeCirculationIfEmpty(CirculationId id);

private:
	//void checkId();

	/// <summary>
	/// Liefert einen Vektor, der die Zuordnung der Servicefahrten zu den Uml�ufen enth�lt.
	/// </summary>
	vector<CirculationId> getServiceTripToCirculationVector(shared_ptr<CuProblem> problem);
		
	int _maxNumOfActions;

	// Daten s�mtlicher Uml�ufe.
	list<pair<CirculationId, Circulation>> _circulations;

	// Circulation-Id f�r den n�chsten neuen Umlauf.
	CirculationId _nextCirculationId;
};
