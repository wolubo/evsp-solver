#include "Solution.h"

#include <tuple>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <algorithm>
#include "EVSP.BaseClasses/Stopwatch.h"
#include "EvspLimits.h"
#include "MatrixCreator.h"
#include "CudaCheck.h"
#include "CuAction.h"
#include "CirculationStats.h"


Solution::Solution(const Solution &other)
	: _maxNumOfActions(other._maxNumOfActions), _circulations(), _nextCirculationId(other._nextCirculationId)
{
	list<pair<CirculationId, Circulation>>::const_iterator iter = other._circulations.begin();
	while (iter != other._circulations.end()) {
		_circulations.push_back(*iter);
		iter++;
	}
}


Solution::Solution(int maxNumOfActions, int numOfServiceTrips)
	: _circulations(), _maxNumOfActions(maxNumOfActions), _nextCirculationId(0)
{
	assert(_maxNumOfActions > 0);
}


Solution::~Solution()
{
}


Solution& Solution::operator=(const Solution &rhs)
{
	if (this != &rhs) {
		_maxNumOfActions = rhs._maxNumOfActions;
		_nextCirculationId = rhs._nextCirculationId;

		_circulations.clear();
		list<pair<CirculationId, Circulation>>::const_iterator iter = rhs._circulations.begin();
		while (iter != rhs._circulations.end()) {
			_circulations.push_back(*iter);
			iter++;
		}
	}
	return *this;
}


bool Solution::operator==(Solution &rhs)
{
	if (this == &rhs) return true;

	if (_maxNumOfActions != rhs._maxNumOfActions) return false;
	if (_circulations.size() != rhs._circulations.size()) return false;

	list<pair<CirculationId, Circulation>>::iterator rhs_iter = rhs._circulations.begin();
	list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
	while (rhs_iter != rhs._circulations.end()) {
		if (iter->second != rhs_iter->second) return false;
		iter++;
		rhs_iter++;
	}

	return true;
}


bool Solution::operator!=(Solution &rhs)
{
	return !(*this == rhs);
}


void Solution::dump()
{
	list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
	while (iter != _circulations.end()) {
		iter->second.dump();
		iter++;
	}
}


bool Solution::check(shared_ptr<CuProblem> problem)
{
	int numOfServiceTrips = problem->getServiceTrips().getNumOfServiceTrips();
	shared_ptr<CuVector1<bool>> ticklist(new CuVector1<bool>(numOfServiceTrips));
	bool retVal = true;

	ticklist->setAll(false);

	if (_circulations.size() == 0) {
		cerr << "Eine gültige Lösung muss Umläufe beinhalten! Diese Lösung ist jedoch leer!" << endl;
		retVal = false;
	}

	shared_ptr<CuVector1<bool>> circIdChecklist(new CuVector1<bool>((short)_nextCirculationId));
	list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
	while (retVal && iter != _circulations.end()) {
		if (circIdChecklist->get((short)iter->first)) {
			cerr << "Die CirculationId " << (short)iter->first << " ist mehrfach belegt!" << endl;
			return false;
		}
		circIdChecklist->set((short)iter->first, true);
		Circulation &circulation = iter->second;
		int numOfActions = circulation.getSize();
		retVal = circulation.check(problem, true, ticklist);
		iter++;
	}

	if (retVal) {
		for (int i = 0; i < numOfServiceTrips; i++) {
			if (!ticklist->get(i)) {
				cerr << "FEHLER: Servicefahrt " << i << " wird nicht bedient." << endl;
			}
		}
	}

	return retVal;
}


SolutionStats Solution::getStats(shared_ptr<CuProblem> problem)
{
	AmountOfMoney totalCost(0);
	float bestCircCostRatio = FLT_MAX;
	float worstCircCostRatio = 0.0f;
	int bestTripLenght = 0;
	int worstTripLenght = INT_MAX;
	int lowestNumOfActions = INT_MAX;
	int highestNumOfActions = 0;

	list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
	while (iter != _circulations.end()) {
		Circulation &currentCirc = iter->second;
		CirculationStats stats = currentCirc.getStats(problem);

		totalCost += stats.totalCost;

		float circCostRatio = stats.getCircCostRatio();
		bestCircCostRatio = std::min(bestCircCostRatio, circCostRatio);
		worstCircCostRatio = std::max(worstCircCostRatio, circCostRatio);

		int numOfServiceTrips = currentCirc.getNumOfActions(CuActionType::SERVICE_TRIP);
		bestTripLenght = std::max(bestTripLenght, numOfServiceTrips);
		worstTripLenght = std::min(worstTripLenght, numOfServiceTrips);

		int numOfActions = currentCirc.getSize();
		lowestNumOfActions = std::min(lowestNumOfActions, numOfActions);
		highestNumOfActions = std::max(highestNumOfActions, numOfActions);

		iter++;
	}

	return SolutionStats(getNumOfCirculations(), totalCost, bestCircCostRatio, worstCircCostRatio, bestTripLenght, worstTripLenght, lowestNumOfActions, highestNumOfActions);
}


int Solution::getNumOfCirculations() const
{
	return (int)_circulations.size();
}


int Solution::getNumOfActions(CirculationId circulationId) const
{
	const Circulation& circ = getCirculation(circulationId);
	return circ.getSize();
}


StopId Solution::getDepotId(CirculationId circulationId) const
{
	const Circulation& circ = getCirculation(circulationId);
	return circ.getDepotId();
}


VehicleTypeId Solution::getVehicleTypeId(CirculationId circulationId)const
{
	const Circulation& circ = getCirculation(circulationId);
	return circ.getVehicleTypeId();
}


CirculationId Solution::createNewCirculation(StopId depotId, VehicleTypeId vehicleTypeId)
{
	return addCirculation(Circulation(depotId, vehicleTypeId, _maxNumOfActions));
}


//void Solution::checkId()
//{
//	shared_ptr<CuVector1<bool>> circIdChecklist(new CuVector1<bool>((short)_nextCirculationId));
//	list<pair<CirculationId, Circulation>>::const_iterator iter = _circulations.begin();
//	while (iter != _circulations.end()) {
//		CirculationId id = iter->first;
//		if (circIdChecklist->get((short)id)) {
//			cerr << "Die CirculationId " << (short)id << " ist mehrfach belegt!" << endl;
//		}
//		circIdChecklist->set((short)id, true);
//		iter++;
//	}
//}


CirculationId Solution::addCirculation(Circulation &newCirculation)
{
	CirculationId circId(_nextCirculationId++);					// Neue Id anlegen
	_circulations.push_back(make_pair(circId, newCirculation));	// Neuen Umlauf anlegen.

	//checkId();

	return circId;
}


CirculationId Solution::createNewCirculation(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem)
{
	DepotTools depotTools(problem);

	const CuServiceTrip &newServiceTrip = problem->getServiceTrips().getServiceTrip(serviceTripId);

	VehicleTypeId  vehicleTypeId = problem->getServiceTripCostMatrix().getBestVehicleType(serviceTripId);
	StopId depotId = depotTools.findBestDepot(newServiceTrip.fromStopId, newServiceTrip.toStopId, vehicleTypeId);
	assert(depotId.isValid());

	Circulation newCirculation = Circulation(depotId, vehicleTypeId, _maxNumOfActions);
	newCirculation.appendServiceTrip(serviceTripId, newServiceTrip, depotId, depotId, problem);

	return addCirculation(newCirculation);
}


void Solution::insertServiceTripGreedy(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem)
{
	int numOfCirculations = (int)_circulations.size();
	if (numOfCirculations > 0) {
		tuple<CirculationId, shared_ptr<Circulation>, AmountOfMoney> theCheapest;
		theCheapest = make_tuple(CirculationId::invalid(), shared_ptr<Circulation>(), AmountOfMoney(INT_MAX));

		// Setze die Servicefahrt probehalber in alle Umläufe ein, die sie aufnehmen können. Merke den kostengünstigsten Umlauf.
		list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
		while (iter != _circulations.end()) {
			Circulation &currCirc = iter->second;
			shared_ptr<Circulation> candidate = currCirc.probeInsertion(serviceTripId, problem);
			if (candidate) {
				AmountOfMoney additionalCost = candidate->getTotalCost(problem);
				AmountOfMoney &cheapest = std::get<2>(theCheapest);
				if (additionalCost < cheapest) {
					cheapest = additionalCost;
					std::get<0>(theCheapest) = iter->first; // Aktuelle CirculationId
					std::get<1>(theCheapest) = candidate;   // Günstigster Umlauf mit neuer Servicefahrt
				}
			}
			iter++;
		}

		shared_ptr<Circulation> theCheapestCirculation = std::get<1>(theCheapest);
		if (theCheapestCirculation) {
			// Ersetze den Umlauf durch die mit probeInsertion() erstellte Kopie.
			CirculationId id = std::get<0>(theCheapest);
			bool done = false;
			iter = _circulations.begin();
			while (!done && iter != _circulations.end()) {
				if (iter->first == id) {
					iter = _circulations.erase(iter);
					iter = _circulations.insert(iter, make_pair(id, *theCheapestCirculation));
					done = true;
				}
				iter++;
			}
		}
		else {
			createNewCirculation(serviceTripId, problem);
		}
	}
	else {
		createNewCirculation(serviceTripId, problem);
	}
}


void Solution::insertServiceTripRandom(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem, RandomCpu &randCpu)
{
	int numOfCirculations = (int)_circulations.size();
	if (numOfCirculations > 0) {
		shared_ptr<Circulation> candidate;
		vector<pair<CirculationId, shared_ptr<Circulation>>> candidates;

		list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
		while (iter != _circulations.end()) {
			Circulation &current = iter->second;
			candidate = current.probeInsertion(serviceTripId, problem);
			if (candidate) {
				candidates.push_back(make_pair(iter->first, candidate));
			}
			iter++;
		}

		int choice = -1;
		if (candidates.size() > 1) {
			choice = randCpu.rand((int)candidates.size());
		}
		else if (candidates.size() == 1) {
			choice = 0;
		}

		if (choice >= 0) {
			// Bisherigen Umlauf durch die mit probeInsertion() erstellte Kopie ersetzen.
			pair<CirculationId, shared_ptr<Circulation>> theChoosenOne = candidates.at(choice);
			bool done = false;
			iter = _circulations.begin();
			while (!done && iter != _circulations.end()) {
				if (iter->first == theChoosenOne.first) {
					iter = _circulations.erase(iter);
					iter = _circulations.insert(iter, make_pair(theChoosenOne.first, *theChoosenOne.second));
					done = true;
				}
				iter++;
			}
		}
		else {
			createNewCirculation(serviceTripId, problem);
		}
	}
	else {
		createNewCirculation(serviceTripId, problem);
	}
}


Circulation& Solution::getCirculation(CirculationId circulationId)
{
	assert(circulationId.isValid());

	list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
	while (iter != _circulations.end()) {
		if (iter->first == circulationId) {
			return iter->second;
		}
		iter++;
	}
	assert(false);
	throw runtime_error("CirculationId ungültig!");
}


const Circulation& Solution::getCirculation(CirculationId circulationId) const
{
	assert(circulationId.isValid());

	list<pair<CirculationId, Circulation>>::const_iterator iter = _circulations.begin();
	while (iter != _circulations.end()) {
		if (iter->first == circulationId) {
			return iter->second;
		}
		iter++;
	}
	assert(false);
	throw runtime_error("CirculationId ungültig!");
}


bool Solution::removeCirculationIfEmpty(CirculationId id)
{
	assert(id.isValid());
	assert(_circulations.size() > 0);
	list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
	while (iter != _circulations.end()) {
		if (iter->first == id) {
			Circulation &toBeChecked = iter->second;
			if (toBeChecked.getNumOfActions(CuActionType::SERVICE_TRIP) == 0) {
				iter = _circulations.erase(iter);
				return true;
			}
		}
		iter++;
	}
	return false;
}


bool removeCirculationVisitor(CuAction& action, vector<ServiceTripId>& leftFreeServiceTrips, int dummy)
{
	if (action.getType() == CuActionType::SERVICE_TRIP) {
		ServiceTripId id = action.getServiceTripId();
		leftFreeServiceTrips.push_back(id);
	}
	return true;
}


vector<ServiceTripId>& Solution::removeCirculation(CirculationId circToBeDeleted, vector<ServiceTripId> &leftFreeServiceTrips)
{
	assert(circToBeDeleted.isValid());

	bool done = false;
	list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
	while (!done && iter != _circulations.end()) {
		if (iter->first == circToBeDeleted) {
			Circulation &circ = iter->second;
			circ.visitItems<vector<ServiceTripId>&, int>(removeCirculationVisitor, leftFreeServiceTrips, 0); // Servicefahrten einsammeln
			iter = _circulations.erase(iter);
			done = true;
		}
		iter++;
	}

	//checkId();

	return leftFreeServiceTrips;
}


bool updateServiceTripToCirculationVisitor(CuAction& action, vector<CirculationId>& serviceTripToCirculation, CirculationId id)
{
	if (action.getType() == CuActionType::SERVICE_TRIP) {
		serviceTripToCirculation.at((short)action.getServiceTripId()) = id;
	}
	return true;
}


vector<CirculationId> Solution::getServiceTripToCirculationVector(shared_ptr<CuProblem> problem)
{
	int numOfServiceTrips = problem->getServiceTrips().getNumOfServiceTrips();
	vector<CirculationId> retVal;
	retVal.reserve(numOfServiceTrips);
	for (int i = 0; i < numOfServiceTrips; i++) retVal.push_back(CirculationId());

	list<pair<CirculationId, Circulation>>::iterator iter = _circulations.begin();
	while (iter != _circulations.end()) {
		CirculationId circId = iter->first;
		Circulation &circ = iter->second;
		circ.visitItems<vector<CirculationId>&, CirculationId>(updateServiceTripToCirculationVisitor, retVal, circId);
		iter++;
	}

	return retVal;
}


void Solution::performCrossoverOperation(int numOfServiceTrips, std::shared_ptr<CuProblem> problem, float crossoverRate, int crossoverUpperBound, RandomCpu &randCpu)
{
	DepotTools depotTools(problem);

	int crossoverLimit = std::min((int)(numOfServiceTrips * crossoverRate), crossoverUpperBound);
	int numOfCrossovers = randCpu.rand(crossoverLimit + 1);

	for (int i = 0; i < numOfCrossovers; i++) {

		// Zuordnungsvektor erzeugen.
		vector<CirculationId> serviceTripToCirculation = getServiceTripToCirculationVector(problem);

		// Zufallsauswahl einer Servicefahrt, die als Splittkriterium dienen wird:
		// Sie definiert den Umlauf A und die Stelle in Umlauf A, an welcher der Kopf 
		// vom Schwanz getrennt wird (nämlich hinter dieser Servicefahrt).
		ServiceTripId circA_lastServTripInHead_id = ServiceTripId(randCpu.rand(numOfServiceTrips));

		//printf("\nSplitt SF: %i\n", (short)circA_lastServTripInHead_id);

		// Umlauf A und Position der Servicefahrt darin finden
		CirculationId circA_id = serviceTripToCirculation.at((short)circA_lastServTripInHead_id);
		assert(circA_id.isValid());
		Circulation& circA = getCirculation(circA_id);
		ItemHandle circA_lastServTripInHead_pos;
		if (circA.gotoServiceTrip(circA_lastServTripInHead_id)) {
			circA_lastServTripInHead_pos = circA.getCurrentPosition();
		}
		else {
			circA.dump();
			assert(false); // Fehler in der Tabelle der ServiceTrip-Zuordnungen?
		}

		// Bestimme die erste Servicefahrt im Schwanz von A.
		ServiceTripId circA_firstServTripInTail_id = circA.gotoNextServiceTripId();
		ItemHandle circA_firstServTripInTail_pos;
		if (circA_firstServTripInTail_id.isValid()) {
			circA_firstServTripInTail_pos = circA.getCurrentPosition();
		}

		// Erzeuge Teilumläufe für den Kopf und den Schwanz von Umlauf A.
		shared_ptr<Circulation> headA;
		if (circA_lastServTripInHead_pos.isValid()) {
			headA = circA.getPartialCirculation(ListRegion(ItemHandle::invalid(), circA_lastServTripInHead_pos, true));
		}

		shared_ptr<Circulation> tailA;
		if (circA_firstServTripInTail_pos.isValid()) {
			tailA = circA.getPartialCirculation(ListRegion(circA_firstServTripInTail_pos, ItemHandle::invalid(), true));
		}

		assert((!tailA
			&& (headA->getNumOfActions(CuActionType::SERVICE_TRIP) + tailA->getNumOfActions(CuActionType::SERVICE_TRIP) == circA.getNumOfActions(CuActionType::SERVICE_TRIP)))
			||
			(!tailA && (headA->getNumOfActions(CuActionType::SERVICE_TRIP) == circA.getNumOfActions(CuActionType::SERVICE_TRIP))));

		// Liste der Servicefahrten zusammenstellen, die potentiell auf die letzte Servicefahrt im Kopf von A folgen können.
		shared_ptr<CuVector1<ServiceTripId>> candidates = problem->getPrevNextMatrix().collectSuccessors(circA_lastServTripInHead_id);

		if (candidates) {

			// Entferne die Servicefahrt, die innerhalb von Umlauf A schon jetzt auf das Splittkriterium folgt aus der Kandidaten-Liste.
			candidates->remove(circA_firstServTripInTail_id);

			int numOfCandidates = candidates->getSize();

			if (numOfCandidates > 0) {

				// Wähle einen Kandidaten aus und prüfe, ob ein Crossover mit dem Umlauf, der den Kandidaten enthält, möglich ist.
				// Tue das, bis ein Crossover erfolgreich durchgeführt werden konnte oder bis kein Kandidat mehr übrig ist.
				while (numOfCandidates > 0) {
					int i = randCpu.rand(numOfCandidates);

					// Umlauf und Position finden
					ServiceTripId circB_firstServTripInTail_id = candidates->get(i);

					//printf("Candidate SF: %i\n", (short)circB_firstServTripInTail_id);

					CirculationId circB_id = serviceTripToCirculation.at((short)circB_firstServTripInTail_id);
					assert(circB_id.isValid()); // PrevNextMatrix oder ServiceTrip-Zuordnung nicht korrekt befüllt!
					Circulation& circB = getCirculation(circB_id);

					ItemHandle circB_firstServTripInTail_pos;
					if (circB.gotoServiceTrip(circB_firstServTripInTail_id)) {
						circB_firstServTripInTail_pos = circB.getCurrentPosition();
					}

					ItemHandle circB_lastServTripInHead_pos;
					ServiceTripId circB_lastServTripInHead_id = circB.getPrevServiceTripId();
					if (circB_lastServTripInHead_id.isValid()) {
						circB_lastServTripInHead_pos = circB.getCurrentPosition();
					}

					// Erzeuge Teilumläufe für den Kopf und den Schwanz von Umlauf B.
					shared_ptr<Circulation> headB;
					if (circB_lastServTripInHead_pos.isValid()) {
						headB = circB.getPartialCirculation(ListRegion(ItemHandle::invalid(), circB_lastServTripInHead_pos, true));
					}

					shared_ptr<Circulation> tailB;
					if (circB_firstServTripInTail_pos.isValid()) {
						tailB = circB.getPartialCirculation(ListRegion(circB_firstServTripInTail_pos, ItemHandle::invalid(), true));
					}

					// Versuche, den Crossover durchzuführen
					shared_ptr<Circulation> newCircA = Circulation::appendCirculation(headA, tailB, problem);
					if (newCircA) {
						shared_ptr<Circulation> newCircB = Circulation::appendCirculation(headB, tailA, problem);
						if (newCircB) {

							//cout << endl;
							//cout << "circA: ";
							//circA.dump();
							//cout << endl;
							//cout << "headA: ";
							//if (headA.isNotNull()) headA->dump();
							//cout << endl;
							//cout << "tailA: ";
							//if (tailA.isNotNull()) tailA->dump();
							//cout << endl;

							//cout << endl;
							//cout << "circB: ";
							//circB.dump();
							//cout << endl;
							//cout << "headB: ";
							//if (headB.isNotNull()) headB->dump();
							//cout << endl;
							//cout << "tailB: ";
							//if (tailB.isNotNull()) tailB->dump();
							//cout << endl;

							// Checke die Aus- und die Einrückfahrten.
							if (depotTools.repairDepot(*newCircA) && depotTools.repairDepot(*newCircB)) {

								// Prüfe die Batteriekapazität.
								if (newCircA->optimizeChargingStationVisits(problem) &&
									newCircB->optimizeChargingStationVisits(problem)) {

									//cout << "***********************" << endl;
									//cout << endl;
									//cout << "VOR CROSSOVER:" << endl;
									//cout << "circA: ";
									//circA.dump();
									//cout << "circB: ";
									//circB.dump();

									// Der Crossover war erfolgreich. Umläufe austauschen. 
									circA = *newCircA;
									removeCirculationIfEmpty(circA_id);

									circB = *newCircB;
									removeCirculationIfEmpty(circB_id);

									//cout << "NACH CROSSOVER:" << endl;
									//cout << "circA: ";
									//circA.dump();
									//cout << "circB: ";
									//circB.dump();

									return;
								}
							}
						}
					}

					candidates->remove(i);
					numOfCandidates--;
				};
			}

		}

		// Es sind keine Kandidaten mehr übrig. Zur ausgewählten Servicefahrt kann kein "Splitt-Point" gefunden werden.
		// Daher wird nun ein neuer Umlauf gebildet, der den Schwanz von Umlauf A aufnimmt.
		if (tailA && tailA->getNumOfActions(CuActionType::SERVICE_TRIP) > 0) {

			//cout << "=======================" << endl;
			//cout << endl;
			//cout << "VOR CROSSOVER:" << endl;
			//cout << "circA: ";
			//circA.dump();

			// Wähle passende Depots und sorge dafür, dass Ein- und Ausrückfahrten vorhanden sind
			if (!depotTools.repairDepot(*headA)) return;
			if (!depotTools.repairDepot(*tailA)) return;

			//cout << "NACH REPAIR DEPOT:" << endl;
			//cout << "headA: ";
			//headA->dump();
			//cout << "circB: ";
			//tailA->dump();

			// Prüfe die Batteriekapazität.
			if (!headA->optimizeChargingStationVisits(problem)) return;
			if (!tailA->optimizeChargingStationVisits(problem)) return;

			assert((!tailA
				&& (headA->getNumOfActions(CuActionType::SERVICE_TRIP) + tailA->getNumOfActions(CuActionType::SERVICE_TRIP) == circA.getNumOfActions(CuActionType::SERVICE_TRIP)))
				||
				(!tailA && (headA->getNumOfActions(CuActionType::SERVICE_TRIP) == circA.getNumOfActions(CuActionType::SERVICE_TRIP))));

			circA = *headA;
			CirculationId tailA_id = addCirculation(*tailA);

			//cout << "NACH CROSSOVER:" << endl;
			//cout << "circA: ";
			//circA.dump();
			//cout << "circB: ";
			//tailA->dump();
		}
	}
}


void Solution::performInsertionOperation(bool greedyInsertion, int numOfServiceTrips, float insertionRate, int insertionUpperBound, shared_ptr<CuProblem> problem, RandomCpu &randCpu)
{
	int insertionLimit = std::min((int)(numOfServiceTrips * insertionRate), insertionUpperBound);
	int numOfInsertions = randCpu.rand(insertionLimit + 1);

	// Erstelle eine Liste mit zufällig ausgewählten Servicefahrten, die aus einem Umlauf entnommen und in einen anderen 
	// Umlauf eingefügt werden sollen.
	ServiceTripId *candidates = new ServiceTripId[numOfInsertions];
	for (int i = 0; i < numOfInsertions; i++) {
		ServiceTripId serviceTripId = ServiceTripId(randCpu.rand(numOfServiceTrips));

		// Schliesse Duplikate aus.
		bool duplicate = false;
		for (int j = 0; j < i; j++) {
			if (candidates[j] == serviceTripId) {
				duplicate = true;
				break;
			}
		}

		if (!duplicate) {
			candidates[i] = serviceTripId;
		}
	}

	// Zuordnungstabelle erzeugen.
	vector<CirculationId> serviceTripToCirculation = getServiceTripToCirculationVector(problem);

	for (int i = 0; i < numOfInsertions; i++) {
		ServiceTripId c = candidates[i];

		if (c.isValid()) {
			// Finde den Umlauf, in dem die Servicefahrt bisher enthalten ist.
			CirculationId circulationId = serviceTripToCirculation.at((short)c);
			assert(circulationId.isValid());
			Circulation& circulation = getCirculation(circulationId);

			// Versuche, die Servicefahrt aus diesem Umlauf zu entfernen.
			if (circulation.removeServiceTrip(c, problem)) {
				removeCirculationIfEmpty(circulationId); // Ist der bisherige Umlauf nun leer? Falls ja kann er entfernt werden.
			}
			else {
				candidates[i] = ServiceTripId::invalid();
			}
		}
	}

	for (int i = 0; i < numOfInsertions; i++) {
		ServiceTripId c = candidates[i];
		if (c.isValid()) {
			// Füge die Servicefahrt in einen andern Umlauf ein.
			if (greedyInsertion)
				insertServiceTripGreedy(c, problem);
			else
				insertServiceTripRandom(c, problem, randCpu);
		}
	}

	delete[] candidates;
}


void Solution::performDeleteOperation(float deletionRate, int lowerBound, int upperBound, DeletionMode deletionMode,
	bool greedyInsertion, shared_ptr<CuProblem> problem, RandomCpu &randCpu)
{
	if (deletionRate < 0.0f) deletionRate = 0.0f;
	if (deletionRate > 1.0f) deletionRate = 1.0f;
	if (lowerBound < 0) lowerBound = 0;
	if (lowerBound > upperBound) lowerBound = upperBound;

	SolutionStats stats = getStats(problem);

	int numOfCirculations = getNumOfCirculations();
	int numOfDeletions = (int)floor(numOfCirculations * deletionRate + 0.5f);
	numOfDeletions = std::max(numOfDeletions, lowerBound);
	numOfDeletions = std::min(numOfDeletions, upperBound);

	if (numOfDeletions == 0) return;

	vector<CirculationId> candidates;
	candidates.reserve(numOfCirculations);
	list<pair<CirculationId, Circulation>>::iterator cand_iter = _circulations.begin();
	while (cand_iter != _circulations.end()) {
		candidates.push_back(cand_iter->first);
		cand_iter++;
	}

	vector<CirculationId> toBeDeleted;
	toBeDeleted.reserve(numOfDeletions);
	for (int i = 0; i < numOfDeletions; i++) {
		int selection;
		if (deletionMode == DeletionMode::CIRC_COST_RATIO) {
			/* Wähle irgendeinen Umlauf aus. Bestimme zu diesem Umlauf die Relation seines Kostenver-        *
			** hältnisses zum besten und zum schlechtesten Kostenverhältnis eines Umlaufs der gesamten       *
			** Lösung. Dieser Wert ist grösser oder gleich 0 und kleiner oder gleich 1. Verwende diesen      *
			** Wert als Wahrscheinlichkeit für die zufallsbasierte Entscheidung darüber, ob der Umlauf       *
			** gelöscht werden soll oder nicht. Wiederhole dies solange, bis die Entscheidung positiv        *
			** ausfällt.                                                                                     *
			** Auf diese Weise werden Umläufe mit einem schlechten Kostenverhältnis mit höherer Wahrschein-   *
			** lichkeit gelöscht als solche mit einem guten Kostenverhältnis.                                */
			float best = stats.getBestCircCostRatio();
			float worst = stats.getWorstCircCostRatio();
			float p; // Wahrscheinlichkeit dafür, dass der Umlauf nicht gelöscht wird.
			do {
				selection = randCpu.rand(numOfCirculations);
				Circulation &circ = getCirculation(candidates.at(selection));
				float circCostRatio = circ.getCircCostRatio(problem);
				p = (worst - circCostRatio) / (worst - best);
				p = powf(p, 2.0f); // Bessere Umläufe übervorteilen.
			} while (randCpu.shot(p));
			toBeDeleted.push_back(candidates.at(selection));
		}
		else if (deletionMode == DeletionMode::NUM_OF_SERVICE_TRIPS) {
			/* Wähle irgendeinen Umlauf aus. Bestimme zu diesem Umlauf die Relation der Anzahl der ent-      *
			** haltenen Servicefahrten zum besten und zum schlechtesten Vergleichswert über alle Umläufe.    *
			** Dieser Wert ist grösser oder gleich 0 und kleiner oder gleich 1. Verwende diesen              *
			** Wert als Wahrscheinlichkeit für die zufallsbasierte Entscheidung darüber, ob der Umlauf       *
			** gelöscht werden soll oder nicht. Wiederhole dies solange, bis die Entscheidung positiv        *
			** ausfällt.                                                                                     *
			** Auf diese Weise werden "kurze" Umläufe, also solche, die nur wenige Servicefahrten bein-      *
			** halten mit höherer Wahrscheinlichkeit gelöscht als solche, die viele Servicefahrten absol-    *
			** vieren.                                                                                       */
			float best = (float)stats.getBestTripLenght();
			float worst = (float)stats.getWorstTripLenght();
			float p; // Wahrscheinlichkeit dafür, dass der Umlauf nicht gelöscht wird.
			do {
				selection = randCpu.rand(numOfCirculations);
				Circulation &circ = getCirculation(candidates.at(selection));
				float numOfServiceTrips = (float)circ.getNumOfActions(CuActionType::SERVICE_TRIP);
				p = (worst - numOfServiceTrips) / (worst - best);
				p = powf(p, 2.0f); // Bessere Umläufe übervorteilen.
			} while (randCpu.shot(p));
			toBeDeleted.push_back(candidates.at(selection));
		}
		else if (deletionMode == DeletionMode::RANDOM) {
			// Wähle den zu löschenden Umlauf rein zufallsbasiert aus.
			selection = randCpu.rand(numOfCirculations);
			toBeDeleted.push_back(candidates.at(selection));
		}
		else {
			assert(false);
		}

		numOfCirculations--;
		if (numOfCirculations > 0 && selection != numOfCirculations) {
			candidates.at(selection) = candidates.at(numOfCirculations);
		}
	}

	vector<ServiceTripId> leftFreeServiceTrips;

	vector<CirculationId>::iterator iter = toBeDeleted.begin();
	while (iter != toBeDeleted.end()) {
		removeCirculation(*iter, leftFreeServiceTrips);
		iter++;
	}

	vector<ServiceTripId>::iterator st_iter = leftFreeServiceTrips.begin();
	while (st_iter != leftFreeServiceTrips.end()) {
		if (greedyInsertion)
			insertServiceTripGreedy(*st_iter, problem);
		else
			insertServiceTripRandom(*st_iter, problem, randCpu);
		st_iter++;
	}
}

