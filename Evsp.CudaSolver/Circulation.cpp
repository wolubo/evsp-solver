#include "Circulation.h"
#include <mutex>
#include <algorithm>
#include "CuAction.h"
#include "DepotTools.h"
#include "CuProblem.h"
#include "CirculationStats.h"
#include "CuServiceTrips.h"
#include "PrevNextMatrix.h"
#include "ConnectionMatrix.h"


Circulation::Circulation(StopId depotId, VehicleTypeId vehicleTypeId, int capacity)
	: Circulation(capacity)
{
	assert(capacity > 0);
	assert(depotId.isValid());
	assert(vehicleTypeId.isValid());

	_vehicleTypeId = vehicleTypeId;

	if (!_vehicleTypeId.isValid()) {
		assert(false);
		throw runtime_error("Fehler: Fahrzeugtyp ist nicht angegeben!");
	}

	_depotId = depotId;

	if (!_depotId.isValid()) {
		assert(false);
		throw runtime_error("Fehler: Depot ist nicht angegeben!");
	}
}


Circulation::Circulation(int capacity)
	: CuDoubleLinkedList1<CuAction>(capacity)
{
	assert(capacity > 0);
}


Circulation::Circulation(const Circulation &other)
	: _depotId(other._depotId), _vehicleTypeId(other._vehicleTypeId), CuDoubleLinkedList1<CuAction>(other)
{
}


Circulation::~Circulation()
{
}


shared_ptr<Circulation> Circulation::merge(Circulation &circA, Circulation &circB, CuDoubleLinkedList1<ServiceTripId> &surplusServiceTrips,
	shared_ptr<CuProblem> problem)
{
	shared_ptr<Circulation> retVal;

	// Passen die Fahrzeugtypen?
	VehicleTypeId vehTypeId;
	VehicleTypeId vehTypeA = circA.getVehicleTypeId();
	VehicleTypeId vehTypeB = circB.getVehicleTypeId();
	if (vehTypeA == vehTypeB) {
		vehTypeId = vehTypeA;
	}
	else if (circB.proofVehicleType(vehTypeA, problem)) { // Kann B auch mit dem Fahrzeugtypen von A bedient werden?
		vehTypeId = vehTypeA;
	}
	else if (circA.proofVehicleType(vehTypeB, problem)) { // Kann A auch mit dem Fahrzeugtypen von B bedient werden?
		vehTypeId = vehTypeB;
	}

	if (vehTypeId.isValid()) {
		DepotTools depotTools(problem);

		// Passen die Depots?
		StopId depotId;
		StopId depotA = circA.getDepotId();
		StopId depotB = circB.getDepotId();
		if (depotA == depotB) {
			depotId = depotA;
		}
		else if (depotTools.proofDepot(circB, depotA)) {
			depotId = depotA;
		}
		else if (depotTools.proofDepot(circA, depotB)) {
			depotId = depotB;
		}

		if (depotId.isValid()) {
			const CuVehicleType &vehType = problem->getVehicleTypes().getVehicleType(vehTypeId);
			int capacity = std::max(circA.getCapacity(), circB.getCapacity());
			shared_ptr<Circulation> newCirculation = make_shared<Circulation>(depotId, vehTypeId, capacity);

			ServiceTripId servTripId = circA.getFirstServiceTripId();
			while (servTripId.isValid()) {
				shared_ptr<Circulation> candidate = newCirculation->probeInsertion(servTripId, problem);
				if (!candidate) {
					surplusServiceTrips.appendItem(servTripId);
				}
				servTripId = circA.gotoNextServiceTripId();
			}

			servTripId = circB.getFirstServiceTripId();
			while (servTripId.isValid()) {
				shared_ptr<Circulation> candidate = newCirculation->probeInsertion(servTripId, problem);
				if (!candidate) {
					surplusServiceTrips.appendItem(servTripId);
				}
				servTripId = circB.gotoNextServiceTripId();
			}

			retVal = newCirculation;
		}
	}

	return retVal;
}


shared_ptr<Circulation> Circulation::getPartialCirculation(const ListRegion &region)
{
	shared_ptr<Circulation> retVal(new Circulation(getCapacity()));
	retVal->_depotId = _depotId;
	retVal->_vehicleTypeId = _vehicleTypeId;
	copy(region, *retVal);
	return retVal;
}


Circulation& Circulation::operator=(const Circulation &rhs)
{
	if (this != &rhs) {
		CuDoubleLinkedList1<CuAction>::operator=(rhs);
		_depotId = rhs._depotId;
		_vehicleTypeId = rhs._vehicleTypeId;
	}
	return *this;
}


bool Circulation::operator==(Circulation &rhs)
{
	if (this == &rhs) return true;

	if (_vehicleTypeId != rhs._vehicleTypeId) return false;
	if (_depotId != rhs._depotId) return false;

	return CuDoubleLinkedList1<CuAction>::operator==(rhs);
}


bool Circulation::operator!=(Circulation &rhs)
{
	return !(*this == rhs);
}


void Circulation::insertEmptyTripBeforeCurrent(StopId from, StopId to, shared_ptr<CuProblem> problem)
{
	if (from != to) {
		CuEmptyTrip emptyTrip;
		EmptyTripId emptyTripId;
		emptyTripId = problem->getConnectionMatrix().getEmptyTripId(from, to);
		assert(emptyTripId.isValid());
		emptyTrip = problem->getEmptyTrips().getEmptyTrip(emptyTripId);

		CuAction emptyTripAction(emptyTrip);
		insertItemBeforeCurrent(emptyTripAction);
	}
}


void Circulation::insertEmptyTripAfterCurrent(StopId from, StopId to, shared_ptr<CuProblem> problem)
{
	if (from != to) {
		CuEmptyTrip emptyTrip;
		EmptyTripId emptyTripId;
		emptyTripId = problem->getConnectionMatrix().getEmptyTripId(from, to);
		assert(emptyTripId.isValid());
		emptyTrip = problem->getEmptyTrips().getEmptyTrip(emptyTripId);

		CuAction emptyTripAction(emptyTrip);
		insertItemAfterCurrent(emptyTripAction);
		//gotoNext();
	}
}


void Circulation::appendEmptyTrip(StopId from, StopId to, shared_ptr<CuProblem> problem)
{
	gotoLast();
	insertEmptyTripAfterCurrent(from, to, problem);
}


void Circulation::appendServiceTrip(ServiceTripId newServiceTripId, const CuServiceTrip &newServiceTrip, StopId from, StopId to, shared_ptr<CuProblem> problem)
{
	assert(from.isValid());
	assert(to.isValid());

	gotoLast();

	CuAction newAction(newServiceTripId, newServiceTrip);
	appendEmptyTrip(from, newServiceTrip.fromStopId, problem);
	insertItemAfterCurrent(newAction);
	appendEmptyTrip(newServiceTrip.toStopId, to, problem);
}


void Circulation::appendCharging(StopId chargingStationId, const CuVehicleType &vehType, shared_ptr<CuProblem> problem)
{
	assert(chargingStationId.isValid());

	gotoLast();

	//const CuStop &chargingStation = problem->getStops().getStop(chargingStationId);
	CuAction newAction(chargingStationId, vehType);
	insertItemAfterCurrent(newAction);
}

mutex errorCheckMutex;

bool Circulation::errorCheck(bool condition, string msg, shared_ptr<CuProblem> problem)
{
	if (!condition) {
		errorCheckMutex.lock();
		cerr << msg << endl;
		dump();
		cerr << endl;
		cerr << endl;
		//dumpTime();
		//cerr << endl;
		dumpBattery(_vehicleTypeId, problem);
		//cerr << endl;
		errorCheckMutex.unlock();
		return true;
	}
	return false;
}


bool Circulation::check(shared_ptr<CuProblem> problem, bool checkCapacity, shared_ptr<CuVector1<bool>> ticklist)
{
	if (errorCheck(problem->getStops().getStop(_depotId).isDepot, "Für diesen Umlauf wurde kein Depot hinterlegt!", problem)) return false;

	if (!isEmpty()) {
		if (errorCheck(first().getFromStopId() == _depotId, "Der Umlauf muss im Depot starten!", problem)) return false;
		if (errorCheck(last().getToStopId() == _depotId, "Der Umlauf muss im Depot enden!", problem)) return false;
	}

	CuVehicleType &vehicle = problem->getVehicleTypes().getVehicleType(_vehicleTypeId);
	if (!check(_depotId, _depotId, _vehicleTypeId, vehicle, problem, checkCapacity, ticklist)) return false;

	return true;
}


bool Circulation::check(StopId departureId, StopId destinationId, VehicleTypeId vehicleTypeId, CuVehicleType &vehicle, shared_ptr<CuProblem> problem, bool checkCapacity, shared_ptr<CuVector1<bool>> ticklist)
{
	if (errorCheck(!isEmpty(), "Dieser Umlauf ist leer.", problem)) return false;

	ItemHandle currentPos = getCurrentPosition();

	StopId currentStop = departureId;
	PointInTime startTime, endTime, currentTime(0);
	bool isFirstServiceTrip = true;
	AmountOfMoney totalCost(0);
	DurationInSeconds totalDuration(0);
	int countActions = 0;
	KilowattHour batteryCapacity = vehicle.batteryCapacity;
	CuActionType prevType = CuActionType::INVALID_ACTION;

	gotoFirst();
	CuAction action = curr();
	startTime = getDepartureTime();
	currentTime = startTime;
	do {
		action = curr();

		if (errorCheck(prevType == CuActionType::SERVICE_TRIP || action.getType() != prevType, "Zwei Aktionen gleichen Typs hintereinander!", problem)) return false;

		switch (action.getType()) {
		case CuActionType::CHARGE:
		{
			StopId chargingStationId = action.getChargingStationId();
			if (errorCheck(chargingStationId == currentStop, "Fehlende Verbindungsfahrt!", problem)) return false;

			const CuStop &stop = problem->getStops().getStop(chargingStationId);
			if (errorCheck(stop.isChargingStation, "Haltestelle ist keine Ladestation!", problem)) return false;

			batteryCapacity = vehicle.batteryCapacity;
			if (errorCheck(action.getDuration() >= vehicle.rechargingTime, "Aufenthalt an der Ladestation ist zu kurz zum Laden!", problem)) return false;
			currentTime = currentTime + (int)action.getDuration();
			//assert(currentTime == action.getEndTime());
			totalDuration += action.getDuration();
			totalCost += vehicle.rechargingCost;
			break;
		}
		case CuActionType::EMPTY_TRIP:
		{
			if (errorCheck(currentStop == action.getFromStopId(), "Verbindungsfahrt mit falscher Starthaltestelle!", problem)) return false;
			EmptyTripId etId = problem->getConnectionMatrix().getEmptyTripId(action.getFromStopId(), action.getToStopId());
			if (etId.isValid()) {
				const CuEmptyTrip &emptyTrip = problem->getEmptyTrips().getEmptyTrip(etId);
				batteryCapacity -= (float)problem->getEmptyTripCostMatrix().getConsumption(etId, vehicleTypeId);
				if (checkCapacity) {
					if (errorCheck(batteryCapacity > 0.0f, "Batteriekapazität reicht nicht aus!", problem)) return false;
				}
				if (errorCheck(action.getDuration() >= emptyTrip.duration, "Zeit reicht nicht für eine Verbindungsfahrt!", problem)) return false;

				totalCost += vehicle.getDistanceDependentCosts(emptyTrip.distance);
				totalDuration += action.getDuration();
				currentStop = action.getToStopId();
				currentTime = currentTime + (int)action.getDuration();
				//assert(currentTime == action.getEndTime());
			}
			else {
				if (errorCheck(false, "Verbindungsfahrt existiert nicht!", problem)) return false;
			}
			break;
		}
		case CuActionType::SERVICE_TRIP:
		{
			if (errorCheck(action.getFromStopId() == currentStop, "Fehlende Verbindungsfahrt!", problem)) return false;
			ServiceTripId stId = action.getServiceTripId();
			if (stId.isValid()) {
				const CuServiceTrip &serviceTrip = problem->getServiceTrips().getServiceTrip(stId);
				if (errorCheck(serviceTrip.fromStopId == action.getFromStopId(), "Starthaltestelle falsch!", problem)) return false;
				if (errorCheck(serviceTrip.toStopId == action.getToStopId(), "Endhaltestelle falsch!", problem)) return false;

				// Ist das Fahrzeugtyp des aktuellen Umlaufs in der Fahrzeugtypgruppe der Servicefahrt der aktuellen Aktion?
				if (errorCheck(problem->getVehicleTypeGroups().hasVehicleType(serviceTrip.vehicleTypeGroupId, vehicleTypeId),
					"Fahrzeugtyp passt nicht!", problem)) return false;

				if (errorCheck(serviceTrip.getDuration() == action.getDuration(), "Dauer der Servicefahrt ist nicht korrekt!", problem)) return false;
				if (errorCheck(currentTime <= serviceTrip.departure, "Aktueller Zeitpunkt liegt nach Abfahrtszeit!", problem)) return false;

				if (isFirstServiceTrip) {
					PointInTime computedStartTime = PointInTime((int)serviceTrip.departure - (int)totalDuration);
					assert(startTime == computedStartTime);
					isFirstServiceTrip = false;
				}
				currentTime = serviceTrip.arrival;
				currentStop = serviceTrip.toStopId;
				batteryCapacity -= (float)problem->getServiceTripCostMatrix().getConsumption(stId, vehicleTypeId);
				if (checkCapacity) {
					if (errorCheck(batteryCapacity > 0.0f, "Batteriekapazität reicht nicht aus!", problem)) return false;
				}
				totalCost += vehicle.getDistanceDependentCosts(serviceTrip.distance);
				totalDuration += action.getDuration();
				if (ticklist) ticklist->set((short)stId, true); // Servicefahrt als "durchgeführt" markieren.
			}
			else {
				if (errorCheck(false, "Die Id der Servicefahrt ist ungültig!", problem)) return false;
			}
			break;
		}
		case CuActionType::VISIT_DEPOT:
		{
			StopId stopId = action.getDepotId();
			if (errorCheck(stopId == currentStop, "Fehlende Verbindungsfahrt!", problem)) return false;

			const CuStop &stop = problem->getStops().getStop(stopId);
			if (errorCheck(stop.isDepot, "Haltestelle ist kein Depot!", problem)) return false;

			currentTime = currentTime + (int)action.getDuration();
			//assert(currentTime == action.getEndTime());
			totalDuration += action.getDuration();
			break;
		}
		default:
			cerr << "Unerwarteter Aktionstyp!" << endl;
			return false;
		}
		prevType = action.getType();
		countActions++;
	} while (gotoNext());

	if (errorCheck(currentStop == destinationId, "Die Zielhaltestelle wird nicht erreicht!", problem)) return false;

	endTime = currentTime;
	if (errorCheck(endTime == getArrivalTime(), "Ende-Zeit nicht korrekt!", problem)) return false;

	CirculationStats stats = getStats(vehicle, ListRegion(), true);

	if (errorCheck(stats.distanceDependentCosts == totalCost, "Streckenabhängige Kosten des Umlaufs nicht korrekt!", problem)) return false;

	DurationInSeconds realTotalDuration((int)endTime - (int)startTime);
	AmountOfMoney timeDependantCosts = vehicle.getTimeDependentCosts(realTotalDuration);
	if (errorCheck(stats.timeDependentCosts == timeDependantCosts, "Zeitabhängige Kosten des Umlaufs nicht korrekt!", problem)) return false;

	totalCost += timeDependantCosts;
	totalCost += vehicle.vehCost;

	if (errorCheck(stats.totalCost == totalCost, "Gesamtkosten des Umlaufs nicht korrekt!", problem)) return false;

	if (errorCheck(stats.totalCost == getTotalCost(vehicle), "getStats() und getTotalCost() liefern unterschiedliche Gesamtkosten!", problem)) return false;

	assert(countActions == getSize());

	setCurrentPosition(currentPos);

	return true;
}


int Circulation::getNumOfActions(CuActionType actionType)
{
	int retVal = 0;
	if (!isEmpty()) {
		ItemHandle pos = getCurrentPosition();
		gotoFirst();
		do {
			if (curr().getType() == actionType) {
				retVal++;
			}
		} while (gotoNext());
		setCurrentPosition(pos);
	}
	return retVal;
}


bool dumpVisitor(CuAction &action, int data1, const int data2)
{
	action.dump();
	printf(" --> ");
	return true;
}


void Circulation::dump(ListRegion region)
{
	if (isEmpty()) return;

	ItemHandle currentPos = getCurrentPosition();

	visitItems<int, int>(dumpVisitor, 0, 0, region);
	printf("\n");

	setCurrentPosition(currentPos);
}


void Circulation::dumpTime()
{
	if (isEmpty()) return;
	ItemHandle currentPos = getCurrentPosition();
	gotoFirst();
	do {
		curr().dumpTime();
		if (hasNext()) printf(" --> ");
	} while (gotoNext());
	printf("\n");
	setCurrentPosition(currentPos);
}


void Circulation::dumpBattery(VehicleTypeId vehTypeId, shared_ptr<CuProblem> problem)
{
	if (isEmpty()) return;
	CuVehicleType &vehicle = problem->getVehicleTypes().getVehicleType(vehTypeId);
	ItemHandle currentPos = getCurrentPosition();
	gotoFirst();
	printf("\nBattery-Dump:\n");
	KilowattHour remaining = vehicle.batteryCapacity;
	printf("Capacity: %f\n", (float)vehicle.batteryCapacity);
	do {
		const CuAction &current = curr();
		printf("Stop: ");
		current.dump();
		KilowattHour consumption(0.0f);
		if (current.getType() == CuActionType::EMPTY_TRIP)
			consumption = vehicle.getEmptyTripConsumption(current.getDistance());
		else if (current.getType() == CuActionType::SERVICE_TRIP)
			consumption = vehicle.getServiceTripConsumption(current.getDistance());
		else if (current.getType() == CuActionType::CHARGE) {
			remaining = vehicle.batteryCapacity;
		}
		remaining -= consumption;
		printf("\nConsumption: %f\n", (float)consumption);
		printf("Remaining: %f\n", (float)remaining);

		ItemHandle pos = getCurrentPosition();
		KilowattHour min = getMinimumCapacity(vehicle, ListRegion(pos, ItemHandle::invalid(), false));
		printf("getMinimumCapacity: %f\n", (float)min);
		KilowattHour rem = getRemainingCapacity(vehicle, ListRegion(ItemHandle::invalid(), pos, true));
		printf("getRemainingCapacity: %f\n", (float)rem);

		if (hasNext()) printf(" --> \n");
	} while (gotoNext());
	printf("\n");
	setCurrentPosition(currentPos);
}


shared_ptr<ListRegion> Circulation::findChargingSlot(KilowattHour remainingCapacity, const CuVehicleType &vehicle, shared_ptr<CuProblem> problem, ListRegion region)
{
	assert(remainingCapacity >= 0.0f);
	ItemHandle left, right;

	shared_ptr<ListRegion> retVal;

	if (region.left.isValid()) {
		setCurrentPosition(region.left);
	}
	else {
		gotoFirst();
	}

	while (curr().getType() != CuActionType::SERVICE_TRIP && curr().getType() != CuActionType::CHARGE && hasNext() && getCurrentPosition() != region.right) {
		if (curr().getType() == CuActionType::EMPTY_TRIP) {
			remainingCapacity -= vehicle.getEmptyTripConsumption(curr().getDistance());
		}
		gotoNext();
	}

	if (curr().getType() == CuActionType::SERVICE_TRIP && !isLast() && remainingCapacity > 0.01f) {
		do {
			remainingCapacity -= vehicle.getServiceTripConsumption(curr().getDistance());
			KilowattHour additionalConsumption(0.0f);
			left = getCurrentPosition();
			const CuAction &leftAction = curr();
			while (gotoNext() && curr().getType() != CuActionType::SERVICE_TRIP && curr().getType() != CuActionType::CHARGE && getCurrentPosition() != region.right) {
				if (curr().getType() == CuActionType::EMPTY_TRIP) {
					additionalConsumption += vehicle.getEmptyTripConsumption(curr().getDistance());
				}
			}
			if (curr().getType() != CuActionType::SERVICE_TRIP) {
				break;
			}
			right = getCurrentPosition();
			const CuAction &rightAction = curr();

			if (remainingCapacity > 0.01f
				&& checkChargingTrip(leftAction.getServiceTripId(), rightAction.getServiceTripId(), ListRegion(left, right), vehicle, remainingCapacity, problem)) {
				if (remainingCapacity > 0.01f) {
					retVal = make_shared<ListRegion>(left, right);
				}
				remainingCapacity -= additionalConsumption;
			}

		} while (!retVal && remainingCapacity > 0.01f);

	}

	return retVal;
}


bool Circulation::checkChargingTrip(ServiceTripId prevId, ServiceTripId nextId, ListRegion region, const CuVehicleType &vehicleType, KilowattHour &remainingCapacity, shared_ptr<CuProblem> problem)
{
	assert(prevId.isValid());
	assert(nextId.isValid());
	assert(remainingCapacity >= 0.0f);

	if (prevId.isValid() && nextId.isValid()) {

		DurationInSeconds maxDuration = problem->getPrevNextMatrix().getInterval(prevId, nextId);

		if (vehicleType.rechargingTime <= maxDuration) {
			// Der Zeitraum zwischen den beiden Servicefahrten reicht grundsätzlich aus, um das Fahrzeug aufzuladen.
			const CuServiceTrip &prev = problem->getServiceTrips().getServiceTrip(prevId);
			const CuServiceTrip &next = problem->getServiceTrips().getServiceTrip(nextId);

			StopId chargingStation = problem->getChargingMatrix().getFastestToReach(prev.toStopId, next.fromStopId);

			if (chargingStation.isValid()) {
				// Es gibt eine Ladestation, die zwischen den beiden Servicefahrten besucht werden kann.
				// TODO Charging-Matrix für Servicefahrten

				shared_ptr<CuEmptyTrip> prevToChargingStationEt = problem->getConnectionMatrix().getEmptyTrip(prev.toStopId, chargingStation);
				assert(prevToChargingStationEt);

				shared_ptr<CuEmptyTrip> chargingStationToNextEt = problem->getConnectionMatrix().getEmptyTrip(chargingStation, next.fromStopId);
				assert(chargingStationToNextEt);

				// Reicht die Zeit auch für die Fahrt zur Ladestation und weiter zur Starthaltestelle der nachfolgenden Servicefahrt?
				if (vehicleType.rechargingTime + prevToChargingStationEt->duration + chargingStationToNextEt->duration <= maxDuration) {

					// Reicht die verfügbare Kapazität für die Fahrt zur Ladestation?
					if (remainingCapacity > vehicleType.getEmptyTripConsumption(prevToChargingStationEt->distance) + 0.01f) {

						// Ermitteln, ob eine Aufladung an dieser Stelle ausreicht, um den Rest des Umlaufs zu absolvieren.
						KilowattHour minRemaining = getMinimumCapacity(vehicleType, ListRegion(region.right, ItemHandle::invalid()));
						KilowattHour etConsumption = vehicleType.getEmptyTripConsumption(chargingStationToNextEt->distance);
						if (vehicleType.batteryCapacity > minRemaining + etConsumption + 0.01f) {
							// Alle Anforderungen werden erfüllt. Es kann also aufgeladen werden.
							remainingCapacity = vehicleType.batteryCapacity - etConsumption;
							return true;
						}
					}
				}
			}
		}
	}

	return false;
}


bool Circulation::getNextServiceTrip(CuAction &result, ItemHandle &handle, bool beginAtFirst)
{
	assert(!isEmpty());

	if (beginAtFirst) {
		gotoFirst();
	}
	else {
		if (!hasNext()) {
			return false;
		}
		gotoNext();
	}

	do {
		const CuAction &current = curr();
		if (current.getType() == CuActionType::SERVICE_TRIP) {
			result = current;
			handle = getCurrentPosition();
			return true;
		}
	} while (gotoNext());

	return false;
}


shared_ptr<Circulation> Circulation::probeInsertion(ServiceTripId newServiceTripId, shared_ptr<CuProblem> problem)
{
	// TODO Kann SF durch Anfahren des Depots erreicht werden? --> Muss schon in PrevNextMatrix berücksichtigt werden!

	assert(_depotId.isValid());
	assert(_vehicleTypeId.isValid());
	assert(newServiceTripId.isValid());

	shared_ptr<Circulation> retVal;

	if (_depotId.isValid() && newServiceTripId.isValid() && _vehicleTypeId.isValid()) {

		const CuServiceTrip &newServiceTrip = problem->getServiceTrips().getServiceTrip(newServiceTripId);

		// Passt der Fahrzeugtyp des Umlaufs zur Fahrzeugtypgruppe der neuen Servicefahrt?
		if (problem->getVehicleTypeGroups().hasVehicleType(newServiceTrip.vehicleTypeGroupId, _vehicleTypeId)) {

			if (isEmpty()) {
				retVal = make_shared<Circulation>(3);
				retVal->appendServiceTrip(newServiceTripId, newServiceTrip, _depotId, _depotId, problem);
			}
			else {
				CuAction leftAction, rightAction;
				ItemHandle leftActionId, rightActionId;
				const PrevNextMatrix &pnMatrix = problem->getPrevNextMatrix();
				const ConnectionMatrix &conMatrix = problem->getConnectionMatrix();

				// Finde die erste Servicefahrt.
				if (getNextServiceTrip(rightAction, rightActionId, true)) {
					if (pnMatrix.checkCombination(newServiceTripId, rightAction.getServiceTripId())) {
						if (!_depotId.isValid() || conMatrix.connected(_depotId, newServiceTrip.fromStopId)) {
							// Die neue Servicefahrt könnte vor der ersten Fahrt stattfinden.
							retVal = make_shared<Circulation>(*this);
							retVal->removeItems(ListRegion(ItemHandle::invalid(), rightActionId, false));
							retVal->setCurrentPosition(rightActionId);
							retVal->insertEmptyTripBeforeCurrent(_depotId, newServiceTrip.fromStopId, problem);
							CuAction newAction(newServiceTripId, newServiceTrip);
							retVal->insertItemBeforeCurrent(newAction);
							retVal->insertEmptyTripBeforeCurrent(newServiceTrip.toStopId, rightAction.getFromStopId(), problem);
						}
					}
					else {
						bool hasNext;
						do {
							leftAction = rightAction;
							leftActionId = rightActionId;
							hasNext = getNextServiceTrip(rightAction, rightActionId);
							if (hasNext) {
								if (pnMatrix.checkCombination(leftAction.getServiceTripId(), newServiceTripId)
									&& pnMatrix.checkCombination(newServiceTripId, rightAction.getServiceTripId()))
								{
									// Die neue Fahrt könnte zwischen zwei bestehenden Fahrten stattfinden.
									retVal = make_shared<Circulation>(*this);
									retVal->removeItems(ListRegion(leftActionId, rightActionId, false));
									retVal->setCurrentPosition(rightActionId);
									retVal->insertEmptyTripBeforeCurrent(leftAction.getToStopId(), newServiceTrip.fromStopId, problem);
									CuAction newAction(newServiceTripId, newServiceTrip);
									retVal->insertItemBeforeCurrent(newAction);
									retVal->insertEmptyTripBeforeCurrent(newServiceTrip.toStopId, rightAction.getFromStopId(), problem);
								}
							}
						} while (!retVal && hasNext);
						if (!retVal) {
							if (pnMatrix.checkCombination(leftAction.getServiceTripId(), newServiceTripId)) {
								if (!conMatrix.connected(newServiceTrip.toStopId, _depotId)) {
									// Die neue Servicefahrt könnte nach der letzten Fahrt stattfinden.
									retVal = make_shared<Circulation>(*this);
									retVal->removeItems(ListRegion(leftActionId, ItemHandle::invalid(), false));
									retVal->setCurrentPosition(leftActionId);
									retVal->insertEmptyTripAfterCurrent(newServiceTrip.toStopId, _depotId, problem);
									CuAction newAction(newServiceTripId, newServiceTrip);
									retVal->insertItemAfterCurrent(newAction);
									retVal->insertEmptyTripAfterCurrent(leftAction.getToStopId(), newServiceTrip.fromStopId, problem);
								}
							}
						}
					}

				}
				else {
					cerr << "Umlauf ohne Servicefahrt:" << endl;
					dump();
					assert(false); // Der Umlauf enthält keine Servicefahrt.
				}
			}

		}
	}

	// Prüfe die Batteriekapazität.
	if (retVal && !retVal->optimizeChargingStationVisits(problem)) {
		retVal.reset();
	}

	return retVal;
}


shared_ptr<Circulation> Circulation::appendCirculation(shared_ptr<Circulation> circA, shared_ptr<Circulation> circB, shared_ptr<CuProblem> problem)
{
	shared_ptr<Circulation> retVal;

	if (circA && circB) {

		ServiceTripId head_lastServiceTrip_id = circA->getLastServiceTripId();
		ServiceTripId tail_firstServiceTrip_id = circB->getFirstServiceTripId();

		// Prüfe, ob die letzte Servicefahrt des Umlaufs ist mit der ersten Servicefahrt des übergebenen Umlaufs kompatibel ist
		if (problem->getPrevNextMatrix().checkCombination(head_lastServiceTrip_id, tail_firstServiceTrip_id)) {

			// Prüfe, ob der sich ergebende Umlauf mit einem der beiden Fahrzeugtypen der (Teil-)Umläufe bedient werden kann
			VehicleTypeId vehTypeId;
			VehicleTypeId vehTypeA = circA->getVehicleTypeId();
			VehicleTypeId vehTypeB = circB->getVehicleTypeId();
			assert(vehTypeA.isValid());
			assert(vehTypeB.isValid());
			if (vehTypeA == vehTypeB) {
				vehTypeId = vehTypeA;
			}
			else if (circB->proofVehicleType(vehTypeA, problem)) { // Kann B auch mit dem Fahrzeugtypen von A bedient werden?
				vehTypeId = vehTypeA;
			}
			else if (circA->proofVehicleType(vehTypeB, problem)) { // Kann A auch mit dem Fahrzeugtypen von B bedient werden?
				vehTypeId = vehTypeB;
			}

			if (vehTypeId.isValid()) {
				// Der sich ergebende Umlauf lässt sich aus einem der beiden Depots der (Teil-)Umläufe bedienen
				DepotTools depotTools(problem);

				StopId depotId;
				StopId depotA = circA->getDepotId();
				StopId depotB = circB->getDepotId();
				assert(depotA.isValid());
				assert(depotB.isValid());
				if (depotA == depotB) {
					depotId = depotA;
				}
				else if (depotTools.proofDepot(*circB, depotA)) {
					depotId = depotA;
				}
				else if (depotTools.proofDepot(*circA, depotB)) {
					depotId = depotB;
				}

				if (depotId.isValid()) {
					// Füge die beiden (Teil-)Umläufe zusammen
					shared_ptr<Circulation> newCirc(new Circulation(*circA));

					// Füge die Verbindungsfahrt hinzu.
					CuAction &lastAction = circA->getLastAction(ListRegion(), CuActionType::SERVICE_TRIP);
					CuAction &firstAction = circB->getFirstAction(ListRegion(), CuActionType::SERVICE_TRIP);
					newCirc->appendEmptyTrip(lastAction.getToStopId(), firstAction.getFromStopId(), problem);

					newCirc->appendList(*circB);

					// Prüfe, ob die Batteriekapazität ausreicht bzw. ob sich eine Aufladung unterbringen lässt
					if (newCirc->optimizeChargingStationVisits(problem)) {
						retVal = newCirc;
					}
				}
			}
		}
	}
	else if (circA) {
		// Falls der Schwanz leer ist: Der Kopf bildet den neuen Umlauf.
		shared_ptr<Circulation> newCirc(new Circulation(*circA));
		newCirc->gotoLast();
		CuAction &lastAction = newCirc->curr();
		if (lastAction.getToStopId() != newCirc->getDepotId()) {
			newCirc->appendEmptyTrip(lastAction.getToStopId(), newCirc->getDepotId(), problem);
		}

		if (newCirc->optimizeChargingStationVisits(problem)) {
			retVal = newCirc;
		}
	}
	else if (circB) {
		// Falls der Kopf leer ist: Der Schwanz bildet den neuen Umlauf.
		shared_ptr<Circulation> newCirc(new Circulation(*circB));
		newCirc->gotoFirst();
		CuAction &firstAction = newCirc->curr();
		if (newCirc->getDepotId() != firstAction.getFromStopId()) {
			shared_ptr<CuEmptyTrip> emptyTrip = problem->getConnectionMatrix().getEmptyTrip(newCirc->getDepotId(), firstAction.getFromStopId());
			CuAction et(*emptyTrip);
			newCirc->insertItemBeforeCurrent(et);
		}

		if (newCirc->optimizeChargingStationVisits(problem)) {
			retVal = newCirc;
		}
	}

	return retVal;
}


struct ProofVehicleTypeVisitorData {
	VehicleTypeId vehType;
	CuProblem *problem;
};

bool proofVehicleTypeVisitor(CuAction &action, bool &retVal, const ProofVehicleTypeVisitorData &data) {
	if (action.getType() == CuActionType::SERVICE_TRIP) {
		const CuServiceTrip& st = data.problem->getServiceTrips().getServiceTrip(action.getServiceTripId());
		if (!data.problem->getVehicleTypeGroups().hasVehicleType(st.vehicleTypeGroupId, data.vehType)) {
			retVal = false;
			return false;
		}
	}
	return true;
}


bool Circulation::proofVehicleType(VehicleTypeId vehType, shared_ptr<CuProblem> problem)
{
	assert(!isEmpty());
	assert(vehType.isValid());
	if (!vehType.isValid()) return false;
	if (isEmpty()) return true;
	if (vehType == _vehicleTypeId) return true;

	ProofVehicleTypeVisitorData data;
	data.vehType = vehType;
	data.problem = problem.get();

	bool retVal = true;
	visitItems<bool&, const ProofVehicleTypeVisitorData&>(proofVehicleTypeVisitor, retVal, data);

	return retVal;
}


bool consumptionCollector(CuAction &action, KilowattHour &consumption, const CuVehicleType& vehType)
{
	switch (action.getType()) {
	case CuActionType::CHARGE:
		return false;
	case CuActionType::EMPTY_TRIP:
		consumption += vehType.getEmptyTripConsumption(action.getDistance());
		break;
	case CuActionType::SERVICE_TRIP:
		consumption += vehType.getServiceTripConsumption(action.getDistance());
		break;
	};

	return true;
}


KilowattHour Circulation::getMinimumCapacity(const CuVehicleType &vehType, ListRegion region)
{
	KilowattHour consumption(0.0f);
	visitItems<KilowattHour&, const CuVehicleType&>(consumptionCollector, consumption, vehType, region);
	return consumption;
}


KilowattHour Circulation::getRemainingCapacity(const CuVehicleType &vehType, ListRegion region)
{
	KilowattHour consumption(0.0f);
	visitItemsReversedOrder<KilowattHour&, const CuVehicleType&>(consumptionCollector, consumption, vehType, region);
	return vehType.batteryCapacity - consumption;
}


bool checkCapacityCollector(CuAction &action, KilowattHour &capacity, const CuVehicleType& vehType)
{
	switch (action.getType()) {
	case CuActionType::CHARGE:
		capacity = vehType.batteryCapacity;
	case CuActionType::EMPTY_TRIP:
		capacity -= vehType.getEmptyTripConsumption(action.getDistance());
		break;
	case CuActionType::SERVICE_TRIP:
		capacity -= vehType.getServiceTripConsumption(action.getDistance());
		break;
	};

	return true;
}


bool Circulation::checkCapacity(const CuVehicleType &vehType)
{
	KilowattHour capacity(vehType.batteryCapacity);
	visitItems<KilowattHour&, const CuVehicleType&>(checkCapacityCollector, capacity, vehType);
	return capacity > 0.0f;
}


bool Circulation::addChargingStationVisitBefore(KilowattHour remainingCapacity, shared_ptr<CuProblem> problem)
{
	if (!hasPrev()) return false;

	const CuVehicleType &vehicle = problem->getVehicleTypes().getVehicleType(_vehicleTypeId);

	// Finde die letzte Servicefahrt vor der aktuellen Aktion, falls die aktuelle Aktion nicht selbst eine Servicefahrt ist.
	CuAction st_right = curr();
	ItemHandle st_right_pos;
	bool go = true;
	do {
		if (curr().getType() == CuActionType::CHARGE) return false;
		if (curr().getType() == CuActionType::SERVICE_TRIP) {
			st_right = curr();
			st_right_pos = getCurrentPosition();
			remainingCapacity += vehicle.getServiceTripConsumption(curr().getDistance());
			go = false;
		}
		else if (curr().getType() == CuActionType::EMPTY_TRIP) {
			remainingCapacity += vehicle.getEmptyTripConsumption(curr().getDistance());
		}
	} while (go && gotoPrev());
	if (go) return false;

	do {
		// Suche eine Servicefahrt, die vor 'st_right' durchgeführt wird.
		CuAction st_left = curr();
		ItemHandle st_left_pos;
		go = true;
		while (go && gotoPrev()) {
			if (curr().getType() == CuActionType::CHARGE) return false;
			if (curr().getType() == CuActionType::SERVICE_TRIP) {
				st_left = curr();
				st_left_pos = getCurrentPosition();
				assert(st_left_pos != st_right_pos);
				remainingCapacity += vehicle.getServiceTripConsumption(curr().getDistance());
				go = false;
			}
			else if (curr().getType() == CuActionType::EMPTY_TRIP) {
				remainingCapacity += vehicle.getEmptyTripConsumption(curr().getDistance());
			}
		}
		if (go) return false;

		// Kann auf dem Weg von der Endhaltestelle von 'st_left' zur Starthaltestelle von 'st_right' eine Ladestation besucht werden?
		DurationInSeconds maxDuration = problem->getPrevNextMatrix().getInterval(st_left.getServiceTripId(), st_right.getServiceTripId());
		assert(maxDuration < 86400);
		maxDuration -= vehicle.rechargingTime;
		if (maxDuration > 0) {
			// Der Zeitraum zwischen den beiden Servicefahrten reicht grundsätzlich aus, um das Fahrzeug aufzuladen.

			// Suche die Ladestation, die am schnellsten zu erreichen ist.
			StopId chargingStation = problem->getChargingMatrix().getFastestToReach(st_left.getToStopId(), st_right.getFromStopId());

			if (chargingStation.isValid()) {
				// Es gibt eine Ladestation, die zwischen den beiden Servicefahrten besucht werden kann.

				// Ermittle die erforderlichen Verbindungsfahrten.
				shared_ptr<CuEmptyTrip> prevToChargingStationEt = problem->getConnectionMatrix().getEmptyTrip(st_left.getToStopId(), chargingStation);
				assert(prevToChargingStationEt);

				KilowattHour consumptionToChargingStation = vehicle.getEmptyTripConsumption(prevToChargingStationEt->distance) + vehicle.getServiceTripConsumption(curr().getDistance());

				// Reicht die Batteriekapazität aus, um die Ladestation zu erreichen?
				if (remainingCapacity > consumptionToChargingStation) {

					// Reicht die Zeit auch für die Fahrt zur Ladestation?
					maxDuration -= prevToChargingStationEt->duration;

					shared_ptr<CuEmptyTrip> chargingStationToNextEt = problem->getConnectionMatrix().getEmptyTrip(chargingStation, st_right.getFromStopId());
					assert(chargingStationToNextEt);

					// Reicht die Zeit auch für die Fahrt von der Ladestation weiter zur Starthaltestelle der nachfolgenden Servicefahrt?
					maxDuration -= chargingStationToNextEt->duration;

					if (maxDuration > 0) {
						// Alle Anforderungen werden erfüllt. Die Ladestation kann also besucht werden.

						// Lösche ggf. alle Aktionen, die bisher zwischen 'st_left' und 'st_right' durchgeführt wurden.
						removeItems(ListRegion(st_left_pos, st_right_pos));

						setCurrentPosition(st_left_pos);

						Circulation chargingTrip(3);
						chargingTrip.appendEmptyTrip(st_left.getToStopId(), chargingStation, problem);
						chargingTrip.appendCharging(chargingStation, vehicle, problem);
						chargingTrip.appendEmptyTrip(chargingStation, st_right.getFromStopId(), problem);

						insertListAfterCurrent(chargingTrip);

						//setCurrentPosition(st_left_pos);

						while (curr().getType() != CuActionType::CHARGE && !isLast()) gotoNext();
						assert(curr().getType() == CuActionType::CHARGE);

						return true;
					}

				}

			}
		}

		st_right = st_left;
		st_right_pos = st_left_pos;

	} while (true);
}


void Circulation::repairEmtpyTrips(shared_ptr<CuProblem> problem)
{
	if (isEmpty()) return;

	const CuVehicleType &vehicle = problem->getVehicleTypes().getVehicleType(_vehicleTypeId);

	StopId lastStopId = _depotId;

	gotoFirst();

	do {
		StopId from = curr().getFromStopId();
		if (from != lastStopId) {
			assert(problem->getConnectionMatrix().connected(lastStopId, from));
			insertEmptyTripBeforeCurrent(lastStopId, from, problem);
		}
		lastStopId = curr().getToStopId();
	} while (gotoNext());

	if (lastStopId != _depotId) {
		assert(problem->getConnectionMatrix().connected(lastStopId, _depotId));
		insertEmptyTripAfterCurrent(lastStopId, _depotId, problem);
	}
}


bool Circulation::removeServiceTrip(ServiceTripId serviceTripId, shared_ptr<CuProblem> problem)
{
	assert(!isEmpty());

	ItemHandle lastPos, nextPos;
	StopId lastStop = _depotId;
	StopId nextStop;

	bool foundIt = false;

	Circulation backup = *this;

	gotoFirst();

	do {
		if (curr().getType() == CuActionType::SERVICE_TRIP) {
			if (foundIt) {
				nextPos = getCurrentPosition();
				nextStop = curr().getFromStopId();
				if (problem->getConnectionMatrix().connected(lastStop, nextStop)) {
					removeItems(ListRegion(lastPos, nextPos, false));
				}
				else {
					foundIt = false;
				}
				break;
			}
			else {
				if (curr().getServiceTripId() == serviceTripId) {
					foundIt = true;
				}
				else {
					lastPos = getCurrentPosition();
					lastStop = curr().getToStopId();
				}
			}
		}
	} while (gotoNext());

	if (foundIt) {
		repairEmtpyTrips(problem);
		if (!optimizeChargingStationVisits(problem)) {
			// Auf Kopie zurücksetzen und FALSE zurück liefern.
			*this = backup;
			foundIt = false;
		}
	}

	return foundIt;
}


bool Circulation::removeChargingStationVisits(shared_ptr<CuProblem> problem)
{
	if (isEmpty()) return true;

	ItemHandle lastServiceTripPos;

	gotoFirst();

	const CuVehicleType &vehicle = problem->getVehicleTypes().getVehicleType(_vehicleTypeId);
	KilowattHour remaining = vehicle.batteryCapacity;
	bool circulationChanged = false;

	do {
		switch (curr().getType()) {
		case CuActionType::SERVICE_TRIP:
			lastServiceTripPos = getCurrentPosition();
			remaining -= vehicle.getServiceTripConsumption(curr().getDistance());
			break;
		case CuActionType::EMPTY_TRIP:
			remaining -= vehicle.getEmptyTripConsumption(curr().getDistance());
			break;
		case CuActionType::CHARGE: {
			assert(lastServiceTripPos.isValid()); // Wieso sollte noch vor dem Absolvieren der ersten Servicefahrt aufgeladen werden?
			while (gotoNext() && curr().getType() != CuActionType::SERVICE_TRIP);
			ItemHandle nextServiceTripPos;
			if (curr().getType() == CuActionType::SERVICE_TRIP) {
				nextServiceTripPos = getCurrentPosition();
			}

			// Lösche die Ladefahrt (also den Besuch der Ladestation und alle etwaigen Verbindungsfahrten zur/von der Ladestation).
			removeItems(ListRegion(lastServiceTripPos, nextServiceTripPos, false));

			lastServiceTripPos = nextServiceTripPos;

			circulationChanged = true;

			break;
		}
		}
	} while (gotoNext());

	bool retVal;

	if (circulationChanged) {
		// Füge die nötigen Verbindungsfahrten hinzu.
		repairEmtpyTrips(problem);
		retVal = checkCapacity(vehicle);
	}
	else {
		retVal = (remaining > 0.0f);
	}

	return retVal;
}


bool Circulation::optimizeChargingStationVisits(shared_ptr<CuProblem> problem)
{
	if (!removeChargingStationVisits(problem)) {

		const CuVehicleType &vehicle = problem->getVehicleTypes().getVehicleType(_vehicleTypeId);

		KilowattHour remaining(vehicle.batteryCapacity);
		gotoFirst();

		do {
			if (curr().getType() == CuActionType::SERVICE_TRIP) {
				remaining -= vehicle.getServiceTripConsumption(curr().getDistance());
			}
			else if (curr().getType() == CuActionType::EMPTY_TRIP) {
				remaining -= vehicle.getEmptyTripConsumption(curr().getDistance());
			}
			else if (curr().getType() == CuActionType::CHARGE) {
				remaining = vehicle.batteryCapacity;
			}

			if (remaining <= 0.01f) {
				//gotoPrev(); // Zurück zur letzten Aktion, die noch bewältigt werden konnte.
				if (addChargingStationVisitBefore(remaining, problem)) {
					remaining = vehicle.batteryCapacity;
				}
				else {
					return false;
				}
			}

			assert(remaining > 0.0f);

		} while (gotoNext());
	}

	return true;
}


bool statsCollector(CuAction &action, CirculationStats &stats, const CuVehicleType &vehType)
{
	const DistanceInMeters &distance = action.getDistance();
	const DurationInSeconds &duration = action.getDuration();

	stats.totalDistance += distance;
	stats.distanceDependentCosts += vehType.getDistanceDependentCosts(distance);

	switch (action.getType()) {
	case CuActionType::CHARGE:
		stats.distanceDependentCosts += vehType.rechargingCost;
		if (stats.minimumCapacity < 0.0f) {
			stats.minimumCapacity = vehType.batteryCapacity - stats.remainingCapacity;
		}
		stats.remainingCapacity = vehType.batteryCapacity;
		break;
	case CuActionType::SERVICE_TRIP:
		stats.remainingCapacity -= vehType.getServiceTripConsumption(distance);
		stats.serviceTripDuration += duration;
		stats.serviceTripCosts += vehType.getTimeDependentCosts(duration) + vehType.getDistanceDependentCosts(distance);
		break;
	case CuActionType::EMPTY_TRIP:
		stats.remainingCapacity -= vehType.getEmptyTripConsumption(distance);
		break;
	}
	return true;
}


CirculationStats Circulation::getStats(CuVehicleType &vehType, ListRegion region, bool addVehicleCost)
{
	CirculationStats retVal;
	if (!isEmpty()) {
		retVal.minimumCapacity = KilowattHour(-1.0f); // Damit im Visitor erkannt wird, dass der Wert noch nicht gesetzt wurde.
		retVal.remainingCapacity = vehType.batteryCapacity;
		visitItems<CirculationStats&, const CuVehicleType&>(statsCollector, retVal, vehType, region);
		retVal.totalDuration = getTime(region).getDuration();
		retVal.timeDependentCosts = vehType.getTimeDependentCosts(retVal.totalDuration);
		retVal.totalCost = retVal.distanceDependentCosts + retVal.timeDependentCosts;
	}
	if (addVehicleCost) {
		retVal.totalCost += vehType.vehCost;
	}
	return retVal;
}


CirculationStats Circulation::getStats(shared_ptr<CuProblem> problem, ListRegion region, bool addVehicleCost)
{
	CuVehicleType &vehType = problem->getVehicleTypes().getVehicleType(getVehicleTypeId());
	return getStats(vehType, region, addVehicleCost);
}


bool totalCostCollector(CuAction &action, AmountOfMoney &totalCost, const CuVehicleType &vehType)
{
	if (action.getType() == CuActionType::CHARGE) totalCost += vehType.rechargingCost;
	totalCost += vehType.getDistanceDependentCosts(action.getDistance());
	return true;
}


AmountOfMoney Circulation::getTotalCost(const CuVehicleType &vehType, ListRegion region, bool addVehicleCost)
{
	AmountOfMoney totalCost(0);
	visitItems<AmountOfMoney&, const CuVehicleType&>(totalCostCollector, totalCost, vehType, region);
	totalCost += vehType.getTimeDependentCosts(getTime().getDuration());
	if (addVehicleCost) {
		totalCost += vehType.vehCost;
	}
	return totalCost;
}


AmountOfMoney Circulation::getTotalCost(shared_ptr<CuProblem> problem, ListRegion region, bool addVehicleCost)
{
	CuVehicleType &vehType = problem->getVehicleTypes().getVehicleType(getVehicleTypeId());
	return getTotalCost(vehType, region, addVehicleCost);
}


float Circulation::getCircCostRatio(shared_ptr<CuProblem> problem)
{
	CirculationStats stats = getStats(problem);
	return stats.getCircCostRatio();
}


struct TimeVisitorData {
	TimeVisitorData() :period(), duration(0) {}
	PeriodOfTime period;
	DurationInSeconds duration;
};


bool timeVisitor(CuAction &action, TimeVisitorData &time, const int dummy)
{
	if (action.getType() == CuActionType::SERVICE_TRIP) {
		time.period = action.getPeriod();
		return false;
	}
	else {
		time.duration += action.getDuration();
	}
	return true;
}


PointInTime Circulation::getDepartureTime(ListRegion region)
{
	TimeVisitorData data;
	visitItems<TimeVisitorData&, int>(timeVisitor, data, 0, region);
	return PointInTime((int)data.period.getBegin() - (int)data.duration);
}


PointInTime Circulation::getArrivalTime(ListRegion region)
{
	TimeVisitorData data;
	visitItemsReversedOrder<TimeVisitorData&, int>(timeVisitor, data, 0, region);
	return PointInTime((int)data.period.getEnd() + (int)data.duration);
}


PeriodOfTime Circulation::getTime(ListRegion region)
{
	return PeriodOfTime(getDepartureTime(region), getArrivalTime(region));
}


bool Circulation::gotoServiceTrip(ServiceTripId servTripId)
{
	bool retVal = false;
	if (gotoFirst()) {
		do {
			if (curr().getType() == CuActionType::SERVICE_TRIP && curr().getServiceTripId() == servTripId) {
				retVal = true;
				break;
			}
		} while (gotoNext());
	}
	return retVal;
}


ServiceTripId Circulation::gotoNextServiceTripId()
{
	while (gotoNext()) {
		if (curr().getType() == CuActionType::SERVICE_TRIP) {
			return curr().getServiceTripId();
		}
	}

	return ServiceTripId::invalid();
}


ServiceTripId Circulation::getPrevServiceTripId()
{
	while (gotoPrev()) {
		if (curr().getType() == CuActionType::SERVICE_TRIP) {
			return curr().getServiceTripId();
		}
	}
	return ServiceTripId::invalid();
}


ServiceTripId Circulation::getFirstServiceTripId()
{
	CuAction action = getFirstAction(ListRegion(), CuActionType::SERVICE_TRIP);
	return action.getServiceTripId();
}


ServiceTripId Circulation::getLastServiceTripId()
{
	CuAction action = getLastAction(ListRegion(), CuActionType::SERVICE_TRIP);
	return action.getServiceTripId();
}


bool getActionVisitor(CuAction &action, CuAction &result, CuActionType actionType)
{
	if (actionType == CuActionType::INVALID_ACTION || action.getType() == actionType) {
		result = action;
		return false;
	}
	return true;
}


CuAction Circulation::getNextAction(ItemHandle pos, CuActionType type)
{
	return getFirstAction(ListRegion(pos, ItemHandle::invalid()), type);
}


CuAction Circulation::getPrevAction(ItemHandle pos, CuActionType type)
{
	return getLastAction(ListRegion(ItemHandle::invalid(), pos), type);
}


CuAction Circulation::getFirstAction(ListRegion region, CuActionType type)
{
	CuAction retVal;
	visitItems<CuAction&, CuActionType>(getActionVisitor, retVal, type, region);
	return retVal;
}


CuAction Circulation::getLastAction(ListRegion region, CuActionType type)
{
	CuAction retVal;
	visitItemsReversedOrder<CuAction&, CuActionType>(getActionVisitor, retVal, type, region);
	return retVal;
}
