#pragma once

#include "EVSP.BaseClasses/Typedefs.h"
#include "EVSP.BaseClasses/PeriodOfTime.hpp"
#include "CuDoubleLinkedList1.hpp"
#include <memory>
#include "CuAction.h"

struct CuServiceTrip;
class PrevNextMatrix;
class ConnectionMatrix;
class CirculationStats;
class CuProblem;



/// <summary>
/// Bildet einen Umlauf bzw. einen Teilumlauf ab.
/// </summary>			
class Circulation : public CuDoubleLinkedList1<CuAction>
{
public:

	Circulation() = delete;

	/// <summary>
	/// Erzeugt einen neuen (leeren) Teilumlauf.
	/// </summary>
	explicit Circulation(int capacity);

	/// <summary>
	/// Erzeugt eine Kopie (deep-copy).
	/// </summary>
	Circulation(const Circulation &other);

	/// <summary>
	/// Erzeugt einen neuen (leeren) Umlauf.
	/// </summary>			
	/// <param name="depotId">Id des Depots, in dem der Umlauf starten und enden soll.</param>
	/// <param name="vehicleTypeId">Id des Fahrzeugtypen, mit dem der Umlauf durchgeführt werden soll.</param>
	/// <param name="capacity">Maximale Anzahl von Aktionen, die der Umlauf aufnehmen kann.</param>
	Circulation(StopId depotId, VehicleTypeId vehicleTypeId, int capacity);

	~Circulation();

	Circulation& operator=(const Circulation &rhs);

	bool operator==(Circulation &rhs);
	bool operator!=(Circulation &rhs);

	/// <summary>
	/// Erzeugt einen neuen Umlauf, indem zwei bestehende Umläufe zusammengeführt werden.
	/// Servicefahrten, die dabei wegen Überschneidungen nicht berücksichtigt werden können werden dem Vektor 'surplusServiceTrips' 
	/// hinzugefügt. 
	/// </summary>			
	/// <param name="circA">Umlauf A</param>
	/// <param name="circB">Umlauf B</param>
	/// <param name="surplusServiceTrips">Sollten Servicefahrten nicht berücksichtigt werden können, werden ihre Ids diesem Vektor hinzugefügt, damit sie später behandelt werden können.</param>
	/// <param name="problem">Wird lediglich für die Initialisierung des Objekts benötigt, aber nicht gespeichert.</param>
	static shared_ptr<Circulation> merge(Circulation &circA, Circulation &circB, CuDoubleLinkedList1<ServiceTripId> &surplusServiceTrips, 
		shared_ptr<CuProblem> problem);

	/// <summary>
	/// Liefert die Anzahl der Aktionen eines bestimmten Typs im Umlauf.
	/// </summary>
	/// <returns></returns>
	int getNumOfActions(CuActionType actionType);

	/// <summary>
	/// Liefert den durch 'slot' definierten Teilumlauf.
	/// </summary>			
	/// <param name="region">Definiert den Bereich des bestehenden Teilumlaufs, der kopiert und zurückgeliefert werden soll.</param>
	shared_ptr<Circulation> getPartialCirculation(const ListRegion &region);

	/// <summary>
	/// Prüft, ob die übergebene Servicefahrt dem Umlauf hinzugefügt werden kann.
	/// Der Energieverbrauch der neuen Servicefahrt wird berücksichtigt und es werden (soweit nötig und möglich), 
	/// Verbindungsfahrten, Auflade-Aktionen Besuche des Depots hinzugefügt.
	/// </summary>			
	/// <param name="serviceTripId">Id einer bisher nicht eingeplanten Servicefahrt.</param>
	/// <param name="problem"></param>
	/// <returns>Im Erfolgsfall wird ein Smartpointer auf einen Clone des Umlaufs geliefert. Falls das Einfügen nicht möglich 
	/// ist wird NULL zurück geliefert.</returns>
	shared_ptr<Circulation> probeInsertion(ServiceTripId newServiceTripId, shared_ptr<CuProblem> problem);

	/// <summary>
	/// Versucht, einen Teilumlauf B an den (Teil-)Umlauf A anzuhängen.
	/// Das gelingt nur, wenn die folgenden Kriterien erfüllbar sind:
	/// - Die letzte Servicefahrt des Umlaufs A ist mit der ersten Servicefahrt des Umlaufs B kompatibel
	/// - Der sich ergebende Umlauf lässt sich aus einem der beiden Depots der (Teil-)Umläufe bedienen
	/// - Der sich ergebende Umlauf lässt sich mit einem der beiden Fahrzeugtypen der (Teil-)Umläufe bedienen
	/// - Die Batteriekapazität reicht aus, um den gesamten Umlauf zu bedienen bzw. es lässt sich eine
	///   Aufladung unterbringen
	/// </summary>			
	/// <param name="circA">(Teil-)Umlauf A</param>
	/// <param name="circB">(Teil-)Umlauf B</param>
	/// <returns>Liefert NULL, falls die Kriterien unerfüllbar sind. Andernfalls wird ein Zeiger auf einen neuen Umlauf geliefert.</returns>
	static shared_ptr<Circulation> appendCirculation(shared_ptr<Circulation> circA, shared_ptr<Circulation> circB, shared_ptr<CuProblem> problem);

	/// <summary>
	/// Liefert die Haltestellen-ID des Depots, in dem der Umlauf startet und endet.
	/// </summary>
	/// <returns></returns>
	StopId getDepotId() const { return _depotId; }

	/// <summary>
	/// Setzt die Haltestellen-ID des Depots, in dem der Umlauf startet und endet.
	/// Es wird dabei nicht geprüft, ob das Depot tatsächlich zum Umlauf passt. Es wird
	/// auch weder eine Ein- noch eine Ausrückfahrt angelegt.
	/// </summary>
	void setDepotId(StopId depotId) { assert(depotId.isValid()); _depotId = depotId; }

	/// <summary>
	/// Liefert die Fahrtzeugtyp-ID des Fahrzeugs, mit dem der Umlauf durchgeführt wird.
	/// </summary>
	/// <returns></returns>
	VehicleTypeId getVehicleTypeId() const { return _vehicleTypeId; }

	/// <summary>
	/// Liefert die Batteriekapazität, die vor Beginn des (Teil-)Umlaufs mindestens vorhanden sein muss, damit der Umlauf erfolgreich 
	/// durchgeführt werden kann. Also den Verbrauch aller Aktionen bis zum Erreichen einer Ladestation bzw. bis zur Rückkehr ins Depot.
	/// </summary>
	/// <param name="region">Bestimmt den Bereich des Umlaufs, der untersucht werden soll. Default: Kompletter Teilumlauf</param>
	/// <param name="vehType">Daten zum verwendeten Fahrzeugtypen.</param>
	/// <returns></returns>
	KilowattHour getMinimumCapacity(const CuVehicleType &vehType, ListRegion region = ListRegion());

	/// <summary>
	/// Liefert die nach Abschluss des (Teil-)Umlaufs noch verfügbare Batteriekapazität.
	/// </summary>
	/// <param name="region">Bestimmt den Bereich des Umlaufs, der untersucht werden soll. Default: Kompletter Teilumlauf</param>
	/// <param name="vehType">Daten zum verwendeten Fahrzeugtypen.</param>
	/// <returns></returns>
	KilowattHour getRemainingCapacity(const CuVehicleType &vehType, ListRegion region = ListRegion());

	/// <summary>
	/// Prüft, ob die verfügbare Batteriekapazität während des gesamten Umlaufs grösser als 0 ist.
	/// </summary>
	bool checkCapacity(const CuVehicleType &vehType);

	/// <summary>
	/// Versucht, auf dem Weg zur aktuellen Aktion den Besuch einer Ladestation einzufügen. Prüft dazu die Lücken zwischen den 
	/// vorhergehenden Servicefahrten. Bricht ab, sobald eine Aktion des Typs CHARGE gefunden wird, da es dann offenbar nicht möglich
	/// ist, zwischen der letzten Aufladung und der aktuellen Aktion erneut aufzuladen.
	/// Im Erfolgsfall wird die neu eingesetzte CHARGE-Aktion zum aktuellen Element.
	/// </summary>			
	/// <param name="remainingCapacity">Die vor der Suche nach einem geeigneten Slot für eine Ladefahrt noch verfügbare Kapazität.
	/// Sie wird benötigt, um zu bestimmen, ob eine Ladestation noch erreicht werden kann. Der Wert kann auch negativ sein.</param>
	/// <returns>TRUE, wenn der Besuch einer Ladestation hinzugefügt werden konnte. Sonst FALSE.</returns>
	bool addChargingStationVisitBefore(KilowattHour remainingCapacity, shared_ptr<CuProblem> problem);

	/// <summary>
	/// Versucht, die Besuche an Ladestationen so zu optimieren, dass möglichst wenige davon nötig sind. Überprüft dabei auch, 
	/// ob die Batteriekapazität ausreicht, um den (Teil-)Umlauf absolvieren zu können und ergänzt - falls nötig - fehlende
	/// Ladefahrten.
	/// </summary>			
	bool optimizeChargingStationVisits(shared_ptr<CuProblem> problem);

	/// <summary>
	/// Liefert statistische Daten zum Umlauf (Batterieverbrauch, Kosten, ...) bezogen auf einen bestimmten Fahrzeugtypen.
	/// </summary>
	/// <param name="vehType">Daten zum verwendeten Fahrzeugtypen.</param>
	/// <param name="region">Definiert den Ausschnitt des Teilumlaufs, zu dem die Daten ermittelt werden sollen. Default: Kompletter Teilumlauf</param>
	/// <param name="addVehicleCost">Steuert, ob die Anschaffungskosten für das verwendete Fahrzeug hinzuaddiert werden sollen oder nicht.</param>
	CirculationStats getStats(CuVehicleType &vehType, ListRegion region = ListRegion(), bool addVehicleCost = true);

	/// <summary>
	/// Liefert statistische Daten zum Umlauf (Batterieverbrauch, Kosten, ...) bezogen auf den im Umlauf hinterlegten Fahrzeugtypen.
	/// </summary>
	/// <param name="region">Definiert den Ausschnitt des Teilumlaufs, zu dem die Daten ermittelt werden sollen. Default: Kompletter Teilumlauf</param>
	/// <param name="addVehicleCost">Steuert, ob die Anschaffungskosten für das verwendete Fahrzeug hinzuaddiert werden sollen oder nicht.</param>
	CirculationStats getStats(shared_ptr<CuProblem> problem, ListRegion region = ListRegion(), bool addVehicleCost = true);

	/// <summary>
	/// Liefert die Gesamtkosten des Umlaufs (ohne Fahrzeuganschaffung) bezogen auf einen bestimmten Fahrzeugtypen.
	/// </summary>
	/// <param name="vehType">Daten zum verwendeten Fahrzeugtypen.</param>
	/// <param name="region">Definiert den Ausschnitt des Teilumlaufs, zu dem die Kosten ermittelt werden sollen. Default: Kompletter Teilumlauf</param>
	AmountOfMoney getTotalCost(const CuVehicleType &vehType, ListRegion region = ListRegion(), bool addVehicleCost = true);

	/// <summary>
	/// Liefert die Gesamtkosten des Umlaufs (ohne Fahrzeuganschaffung) bezogen auf den im Umlauf hinterlegten Fahrzeugtypen.
	/// </summary>
	/// <param name="vehType">Daten zum verwendeten Fahrzeugtypen.</param>
	/// <param name="region">Definiert den Ausschnitt des Teilumlaufs, zu dem die Kosten ermittelt werden sollen. Default: Kompletter Teilumlauf</param>
	AmountOfMoney getTotalCost(shared_ptr<CuProblem> problem, ListRegion region = ListRegion(), bool addVehicleCost = true);

	/// <summary>
	/// Liefert das Brutto-/Netto-Kostenverhältnis des Umlaufs. 
	/// Das Kostenverhältnis eines Umlaufs gibt das Verhältnis der Durchführungskosten (inkl. aller anderen Kosten wie Wartezeiten, 
	/// Fahrzeuganschaffung, Aufladungen und Leerfahrten) zu den Kosten, die für die reine Durchführung der Servicefahrten entstehen 
	/// würden wieder. Letztere sind ein Idealwert, der in der Praxis nicht erreichbar sein wird, weil ja zumindest für die 
	/// Anschaffung des Fahrzeugs und für unvermeidliche Wartezeiten Kosten anfallen.
	/// Der beste erreichbare Wert ist 1.0. Höhere Werte bedeuten eine schlechtere Effizienz des betreffenden Umlaufs (sub-optimaler 
	/// Einsatz der Betriebsmittel).
	/// </summary>
	float getCircCostRatio(shared_ptr<CuProblem> problem);

	/// <summary>
	/// Liefert den Zeitraum, in dem der (Teil-)Umlauf stattfindet. Also den Zeitraum vom Beginn der Ausrückfahrt bis zur Ankunft der Einrückfahrt.
	/// <param name="region">Definiert den Ausschnitt des Teilumlaufs, zu dem der Zeitraum ermittelt werden sollen. Default: Kompletter Teilumlauf</param>
	/// </summary>
	PeriodOfTime getTime(ListRegion region = ListRegion());

	/// <summary>
	/// Liefert den Zeitpunkt, zu dem der (Teil-)Umlauf startet.
	/// <param name="region">Definiert den Ausschnitt des Teilumlaufs, zu dem der Zeitpunkt ermittelt werden sollen. Default: Kompletter Teilumlauf</param>
	/// </summary>
	PointInTime getDepartureTime(ListRegion region = ListRegion());

	/// <summary>
	/// Liefert den Zeitpunkt, zu dem der (Teil-)Umlauf endet.
	/// <param name="region">Definiert den Ausschnitt des Teilumlaufs, zu dem der Zeitpunkt ermittelt werden sollen. Default: Kompletter Teilumlauf</param>
	/// </summary>
	PointInTime getArrivalTime(ListRegion region = ListRegion());

	/// <summary>
	/// Prüft, ob der Umlauf korrekt ist. Ob er also lückenlos ist und ob alle Aktionen durchführbar sind.
	/// </summary>			
	bool check(shared_ptr<CuProblem> problem, bool checkCapacity = true, shared_ptr<CuVector1<bool>> ticklist = 0);

	/// <summary>
	/// Prüft, ob der Teilumlauf korrekt ist. Ob er also lückenlos ist und ob alle Aktionen durchführbar sind.
	/// </summary>			
	bool check(StopId departureId, StopId destinationId, VehicleTypeId vehicleTypeId, CuVehicleType &vehicle, shared_ptr<CuProblem> problem, bool checkCapacity = true, shared_ptr<CuVector1<bool>> ticklist = 0);

	/// <summary>
	/// Liefert das Handle der Aktion, die für eine bestimmte Servicefahrt steht.
	/// </summary>			
	bool gotoServiceTrip(ServiceTripId servTripId);

	/// <summary>
	/// Liefert die Id der nächsten Servicefahrt.
	/// Dabei wird von der gerade aktuellen Aktion ausgegangen.
	/// </summary>			
	ServiceTripId gotoNextServiceTripId();

	/// <summary>
	/// Liefert die Id der vorhergehenden Servicefahrt.
	/// Dabei wird von der gerade aktuellen Aktion ausgegangen.
	/// </summary>			
	ServiceTripId getPrevServiceTripId();

	/// <summary>
	/// Liefert die Id der ersten Servicefahrt innerhalb des durch 'region' definierten Teilumlaufs.
	/// </summary>			
	ServiceTripId getFirstServiceTripId();

	/// <summary>
	/// Liefert die Id der letzten Servicefahrt innerhalb des durch 'region' definierten Teilumlaufs.
	/// </summary>			
	ServiceTripId getLastServiceTripId();

	/// <summary>
	/// Liefert die nächste Aktion des Typs 'type', welche auf die durch 'pos' definierte Aktion folgt. 
	/// </summary>			
	CuAction getNextAction(ItemHandle pos, CuActionType type);

	/// <summary>
	/// Liefert die Aktion des Typs 'type', die der durch 'pos' definierten Aktion vorhergeht. 
	/// </summary>			
	CuAction getPrevAction(ItemHandle pos, CuActionType type);

	/// <summary>
	/// Liefert die erste Aktion des Typs 'type' innerhalb des durch 'region' definierten Teilumlaufs.
	/// Die Aktion wird zur aktuellen Aktion, auf die sich nachfolgende Operationen beziehen.
	/// Ist 'type' gleich INVALID_ACTION wird der Type nicht berücksichtigt.
	/// </summary>			
	CuAction getFirstAction(ListRegion region = ListRegion(), CuActionType type = CuActionType::INVALID_ACTION);

	/// <summary>
	/// Liefert die letzte Aktion des Typs 'type' innerhalb des durch 'region' definierten Teilumlaufs.
	/// Die Aktion wird zur aktuellen Aktion, auf die sich nachfolgende Operationen beziehen.
	/// Ist 'type' gleich INVALID_ACTION wird der Type nicht berücksichtigt.
	/// </summary>			
	CuAction getLastAction(ListRegion region = ListRegion(), CuActionType type = CuActionType::INVALID_ACTION);

	/// <summary>
	/// Prüft, ob sich der Umlauf bzw. der Teilumlauf mit einem bestimmten Fahrzeugtypen durchführen lässt.
	/// </summary>			
	bool proofVehicleType(VehicleTypeId vehType, shared_ptr<CuProblem> problem);

	void dump(ListRegion region = ListRegion());
	void dumpTime();
	void dumpBattery(VehicleTypeId vehTypeId, shared_ptr<CuProblem> problem);

	/// <summary>
	/// Sucht innerhalb des Teilumlaufs nach einer Lücke zwischen zwei Servicefahrten, in der eine Ladestation angefahren werden kann.
	/// </summary>			
	/// <param name="remainingCapacity">Enthält die zu Begin des durch 'region' definierten Bereichs noch verfügbare Batteriekapazität.</param>
	/// <param name="region">Bereich des Umlaufs, in dem nach einer Auflademöglichkeit gesucht werden soll.</param>
	/// <returns>Region, welche die Daten der gefundenen Lücke enthält oder NULL</returns>
	shared_ptr<ListRegion> findChargingSlot(KilowattHour remainingCapacity, const CuVehicleType &vehicle, shared_ptr<CuProblem> problem, ListRegion region = ListRegion());

	/// <summary>
	/// Ermittelt, ob das Aufladen eines bestimmten Fahrzeugs zwischen zwei bestimmten Servicefahrten
	/// mit gegebener Batteriekapazität möglich ist. Liefert im Erfolgsfall die Daten der Ladefahrt.
	/// </summary>
	/// <param name="prev">Id der Starthaltestelle</param>
	/// <param name="next">Id der Zielhaltestelle</param>
	/// <param name="region">ListRegion, die ins Ergebnis eingetragen werden soll, falls das Aufladen möglich ist.</param>
	/// <param name="vehicleType">Daten des Fahrzeugtyps</param>
	/// <param name="remainingCapacity">Vor Aufruf: Die vor Antritt der Ladefahrt noch verfügbare Batteriekapazität. Nach Aufruf: Restkapazität nach Ladefahrt.</param>
	/// <returns>TRUE, wenn das Aufladen zwischen den beiden Servicefahrten möglich ist. Sonst FALSE.</returns>
	bool checkChargingTrip(ServiceTripId prevId, ServiceTripId nextId, ListRegion region, const CuVehicleType &vehicleType, KilowattHour &remainingCapacity, shared_ptr<CuProblem> problem);

	/// <summary>
	/// 
	/// </summary>
	/// <param name=""></param>
	/// <returns></returns>
	void insertEmptyTripBeforeCurrent(StopId from, StopId to, shared_ptr<CuProblem> problem);

	/// <summary>
	/// 
	/// </summary>
	/// <param name=""></param>
	/// <returns></returns>
	void insertEmptyTripAfterCurrent(StopId from, StopId to, shared_ptr<CuProblem> problem);

	/// <summary>
	/// 
	/// </summary>
	/// <param name=""></param>
	/// <returns></returns>
	void appendEmptyTrip(StopId from, StopId to, shared_ptr<CuProblem> problem);

	/// <summary>
	/// 
	/// </summary>
	/// <param name=""></param>
	/// <returns></returns>
	void appendServiceTrip(ServiceTripId newServiceTripId, const CuServiceTrip &newServiceTrip, StopId from, StopId to, shared_ptr<CuProblem>);

	/// <summary>
	/// 
	/// </summary>
	/// <param name=""></param>
	/// <returns></returns>
	void appendCharging(StopId chargingStationId, const CuVehicleType &vehType, shared_ptr<CuProblem> problem);

	bool errorCheck(bool condition, string msg, shared_ptr<CuProblem> problem);

	/// <summary>
	/// Liefert die nächste Aktion vom Typen CuActionType::SERVICE_TRIP und macht sie innerhalb von _actions zum aktuellen Element.
	/// Zusätzlich werden die Summe der Kosten und die Summe des Verbrauchs aller übersprungenen Aktionen geliefert.
	/// </summary>
	/// <param name="result">Nimmt im Erfolgsfall das Ergebnis auf.</param>
	/// <param name="handle">Nimmt im Erfolgsfall das Handle auf das Servicefahrt-Item auf.</param>
	/// <param name="beginAtFirst">TRUE: Beginne die Suche beim ersten Element.</param>
	/// <returns>Liefert TRUE, wenn auf das derzeit aktuelle (bzw. das erste) Element noch ein Element vom Typen CuActionType::SERVICE_TRIP folgt. 
	/// Andernfalls FALSE.</returns>
	bool getNextServiceTrip(CuAction &result, ItemHandle &handle, bool beginAtFirst = false);

	/// <summary>
	/// Ergänzt fehlende Verbindungsfahrten.
	/// Dabei wird davon ausgegangen, dass der Umlauf prinzipiell korrekt ist und dass lediglich Verbindungsfahrten fehlen, weil 
	/// Änderungen am Umlauf vorgenommen wurden. Dabei wird die Batteriekapazität nicht berücksichtigt! Ggf. müssen im Anschluss
	/// noch Aufladungen ergänzt werden.
	/// </summary>			
	void repairEmtpyTrips(shared_ptr<CuProblem> problem);

	/// <summary>
	/// Entfernt eine Servicefahrt aus dem Umlauf. Es entsteht dabei möglicherweise ein leerer Umlauf!
	/// </summary>			
	/// <returns>Liefert TRUE, wenn das Entfernen der Servicefahrt möglich ist.</returns>
	bool removeServiceTrip(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem);

private:
	/// <summary>
	/// Entfernt alle etwaigen Aufladungen. 
	/// </summary>			
	/// <returns>Liefert TRUE, wenn der (Teil-)Umlauf auch nach dem Entfernen aller Ladestationen noch gültig ist.</returns>
	bool removeChargingStationVisits(shared_ptr<CuProblem> problem);

	/// <summary>
	/// Id des Fahrzeugtypen, mit dem der Teilumlauf durchgeführt wird.
	/// </summary>			
	VehicleTypeId _vehicleTypeId;

	/// <summary>
	/// Haltestellen-Id des Depots.
	/// </summary>			
	StopId _depotId;
};


