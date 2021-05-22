#pragma once

#include "EVSP.BaseClasses/DeviceLaunchParameters.h"

#include "Solution.h"


class CuSolution
{
public:
	static CuSolution* createOnDevice(int initialCapacity, int growRate);
	static void deleteOnDevice(CuSolution** devicePtr);
	static CuSolution* copy2host(CuSolution* devicePtr);

	Solution* getCirculation(int index);
	void addCirculation(Solution* newCirculation);
	void removeCirculation(int index);

	//void setNumOfServiceTrips(int value) { _numOfServiceTrips = value; }
	int getNumOfServiceTrips() { return _numOfServiceTrips; }

	//void setNumOfEmptyTrips(int value) { _numOfEmptyTrips = value; }
	int getNumOfEmptyTrips() { return _numOfEmptyTrips; }

	//void setNumOfChargingEvents(int value) { _numOfChargingEvents = value; }
	int getNumOfChargingEvents() { return _numOfChargingEvents; }

	//void setTotalCost(int value) { _totalCost = value; }
	int getTotalCost() { return _totalCost; }

	//void setTotalKmCost(int value) { _totalKmCost = value; }
	int getTotalKmCost() { return _totalKmCost; }

	//void setTotalHourCost(int value) { _totalHourCost = value; }
	int getTotalHourCost() { return _totalHourCost; }

	//void setTotalVehicleCost(int value) { _totalVehicleCost = value; }
	int getTotalVehicleCost() { return _totalVehicleCost; }

	//void setCirculations(Solution* circulations, int numOfcirculations);
	int getNumOfCirculations() { return _numOfCirculations; }
	Solution** getCirculations() { return _circulations; }

private:
	int _numOfServiceTrips;
	int _numOfEmptyTrips;
	int _numOfChargingEvents;
	int _totalCost;
	int _totalKmCost;
	int _totalHourCost;
	int _totalVehicleCost;
	int _numOfCirculations;

	Solution **_circulations;
	int _capacity;
	int _growRate;

	CuSolution() {}
	CuSolution(CuSolution&) = delete;
	~CuSolution() {}
};

