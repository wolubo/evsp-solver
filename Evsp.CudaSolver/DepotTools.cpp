#include "DepotTools.h"


DepotTools::DepotTools(shared_ptr<CuProblem> problem)
	: _problem(problem)
{
}


DepotTools::~DepotTools()
{
}


bool DepotTools::repairDepot(Circulation &circ)
{
	assert(!circ.isEmpty());
	if (circ.isEmpty()) return false;

	ItemHandle pos = circ.getCurrentPosition();

	// Depot nicht definiert? Finde ein passendes Depot.
	StopId depotId = circ.getDepotId();
	if (!depotId.isValid()) {
		depotId = findBestDepot(circ);
		if (!depotId.isValid()) return false;
		circ.setDepotId(depotId);
	}
	else {
		// Startet der Umlauf im Depot?
		circ.gotoFirst();
		StopId startStopId = circ.curr().getFromStopId();
		if (startStopId != depotId) {
			// Fehlende oder falsche Ausrückfahrt
			if (circ.curr().getType() == CuActionType::SERVICE_TRIP) {
				circ.insertEmptyTripBeforeCurrent(depotId, startStopId, _problem);
			}
			else {
				assert(false);
				return false;
			}
		}

		circ.gotoLast();
		StopId endStopId = circ.curr().getToStopId();
		if (endStopId != depotId) {
			// Fehlende oder falsche Einrückfahrt
			if (circ.curr().getType() == CuActionType::SERVICE_TRIP) {
				circ.insertEmptyTripAfterCurrent(endStopId, depotId, _problem);
			}
			else {
				assert(false);
				return false;
			}
		}
	}

	circ.setCurrentPosition(pos);

	return true;
}


CU_HSTDEV StopId DepotTools::findBestDepot(StopId start, StopId end, VehicleTypeId vehicleTypeId) const
{
	assert(start.isValid());
	assert(end.isValid());
	assert(vehicleTypeId.isValid());

	StopId bestDepot;
	if (start.isValid() && end.isValid() && vehicleTypeId.isValid()) {
		AmountOfMoney tc, btc(INT_MAX);
		StopId aDepot;
		EmptyTripId etId;
		const CuStops &stops = _problem->getStops();
		const ConnectionMatrix &connectionMatrix = _problem->getConnectionMatrix();
		const EmptyTripCostMatrix &etCostMatrix = _problem->getEmptyTripCostMatrix();

		// Berechne für jedes Depot die Summe der Kosten für die Einrück- und die Ausrückfahrt. 
		// Finde das Depot, bei dem diese Kosten am geringsten sind.
		for (DepotId depotIdx(0); depotIdx < stops.getNumOfDepots(); depotIdx++) {
			aDepot = stops.getStopIdOfDepot(depotIdx);
			etId = connectionMatrix.getEmptyTripId(aDepot, start);
			if (etId.isValid()) {
				tc = etCostMatrix.getTotalCost(etId, vehicleTypeId);
				etId = connectionMatrix.getEmptyTripId(end, aDepot);
				if (etId.isValid()) {
					tc = tc + etCostMatrix.getTotalCost(etId, vehicleTypeId);
					if (tc < btc) {
						btc = tc;
						bestDepot = aDepot;
					}
				}
			}
		}
	}

	return bestDepot;
}


CU_HSTDEV StopId DepotTools::findBestDepot(Circulation &circ) const
{
	VehicleTypeId vehicleTypeId = circ.getVehicleTypeId();
	assert(vehicleTypeId.isValid());

	const CuAction &firstAction = circ.getFirstAction();
	StopId startStopId = firstAction.getFromStopId();

	const CuAction &lastAction = circ.getLastAction();
	StopId endStopId = lastAction.getToStopId();

	return findBestDepot(startStopId, endStopId, vehicleTypeId);
}


CU_HSTDEV StopId DepotTools::findRandomDepot(StopId start, StopId end, StopId exclude, int randomNumber) const
{
	assert(start.isValid());
	assert(end.isValid());
	assert(randomNumber >= 0);

	StopId randomDepot;
	if (start.isValid() && end.isValid()) {
		StopId aDepot;
		EmptyTripId etId;
		const CuStops &stops = _problem->getStops();
		const ConnectionMatrix &connectionMatrix = _problem->getConnectionMatrix();

		int numOfCandidates = 0;
		int numOfDepots = stops.getNumOfDepots();
		StopId *candidates = new StopId[numOfDepots];

		for (DepotId depotIdx(0); depotIdx < numOfDepots; depotIdx++) {
			aDepot = stops.getStopIdOfDepot(depotIdx);
			if (aDepot == exclude) continue;
			etId = connectionMatrix.getEmptyTripId(aDepot, start);
			if (etId.isValid()) {
				etId = connectionMatrix.getEmptyTripId(end, aDepot);
				if (etId.isValid()) {
					candidates[numOfCandidates++] = aDepot;
				}
			}
		}

		if (numOfCandidates > 1) {
			randomDepot = candidates[randomNumber % numOfDepots];
		}
		else if (numOfCandidates == 1) {
			randomDepot = candidates[0];
		}

		delete[] candidates;
	}

	return randomDepot;
}


CU_HSTDEV StopId DepotTools::findRandomDepot(Circulation &circ, int randomNumber) const
{
	assert(randomNumber >= 0);

	const CuAction &firstAction = circ.getFirstAction();
	StopId startStopId = firstAction.getFromStopId();

	const CuAction &lastAction = circ.getLastAction();
	StopId endStopId = lastAction.getToStopId();

	return findRandomDepot(startStopId, endStopId, circ.getDepotId(), randomNumber);
}


bool DepotTools::proofDepot(Circulation &circ, StopId depotId)
{
	assert(depotId.isValid());
	if (!depotId.isValid()) return false;

	if (circ.getNumOfActions(CuActionType::SERVICE_TRIP) == 0)
	{
		assert(false);
		return false;
	}

	const ConnectionMatrix &connectionMatrix = _problem->getConnectionMatrix();
	ServiceTripId serviceTripId = circ.getFirstServiceTripId();
	CuServiceTrip serviceTrip = _problem->getServiceTrips().getServiceTrip(serviceTripId);
	if (connectionMatrix.connected(depotId, serviceTrip.fromStopId)) {
		serviceTripId = circ.getLastServiceTripId();
		serviceTrip = _problem->getServiceTrips().getServiceTrip(serviceTripId);
		return connectionMatrix.connected(serviceTrip.fromStopId, depotId);
	}

	return false;
}


