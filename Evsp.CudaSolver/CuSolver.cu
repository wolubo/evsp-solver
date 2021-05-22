#include "CuSolver.h"

#include "cuda_runtime.h"

#include "EVSP.Model/Stop.h"
#include "EVSP.Model/EmptyTrip.h"

#include "CuProblem.h"
#include "CuSolution.h"



bool CuSolver::setupCuda()
{
	bool retVal = false;
	int nDevices;
	cudaError_t result = cudaGetDeviceCount(&nDevices);
	if (result == cudaSuccess && nDevices > 0) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		printf("CUDA-konforme Grafikkarte gefunden: %s\n", prop.name);
		CUDA_CHECK(cudaSetDevice(0));
		retVal = true;
	}
	else {
		printf("Keine CUDA-konforme Grafikkarte vorhanden! Berechnungen werden auf der CPU durchgeführt!\n");
	}
	return retVal;
}


void CuSolver::resetCuda()
{
	CUDA_CHECK(cudaDeviceReset());
}


CuSolver::CuSolver(shared_ptr<ResultLogger> resultLogger)
	: _terminationDelegate(0), _maxNumOfRounds(1000), _maxElapsedTime(1000), _stopwatch(), _roundCounter(0),
	_resultLogger(resultLogger)
{
}


CuSolver::~CuSolver()
{
}


void CuSolver::addResult(string caption, AmountOfMoney totalCost, int numOfVehicles)
{
	_resultLogger->addEntry(caption, _roundCounter, _stopwatch.elapsedSeconds(), totalCost, numOfVehicles);
}


void CuSolver::setTerminationDelegate(TerminationDelegate terminationDelegate)
{
	_terminationDelegate = terminationDelegate;
}


void CuSolver::setMaxNumOfRounds(int maxNumOfRounds)
{
	_maxNumOfRounds = maxNumOfRounds;
}


void CuSolver::setMaxElapsedTime(float maxElapsedTime)
{
	_maxElapsedTime = maxElapsedTime;
}


bool CuSolver::checkTerminationConditions()
{
	bool terminate = (_maxNumOfRounds > 0 && _roundCounter >= _maxNumOfRounds);
	if (!terminate & _maxElapsedTime > 0.0f) {
		terminate = terminate || _stopwatch.elapsedSeconds() > _maxElapsedTime;
	}
	if (!terminate && _terminationDelegate != 0) {
		terminate = _terminationDelegate();
	}
	return terminate;
}


void CuSolver::increaseRoundCounter()
{
	_roundCounter++;
}


void CuSolver::run()
{
	_stopwatch.start();
	_roundCounter = 0;
	setup();
	_roundCounter = 1;
	do {
		loop();
		increaseRoundCounter();
	} while (!checkTerminationConditions());
	teardown();
}


float CuSolver::getElapsedSeconds()
{
	return _stopwatch.elapsedSeconds();
}


std::string CuSolver::getElapsedSecondsAsStr(int numOfDigits)
{
	return _stopwatch.elapsedSecondsAsStr(numOfDigits);
}
