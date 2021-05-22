#include "CuSolutionGeneratorAcoGpu.h"

#include <thread>
#include "Temperature.hpp"
#include "cuda_runtime.h"
#include "EVSP.BaseClasses/StopWatch.h"
#include "CuBoolVector1.hpp"
#include "RandomGpu.h"
#include "AntsRaceKernel.h"


CuSolutionGeneratorAcoGpu::CuSolutionGeneratorAcoGpu(std::shared_ptr<CuProblem> problem, std::shared_ptr<CuConstructionGraph> ptn, int populationSize,	int maxNumOfCirculations, int maxNumOfNodes, float chargeLevel, shared_ptr<RandomGpu> rand, bool verbose, bool keepBestSolution)
	: CuSolutionGenerator(problem, ptn, populationSize, maxNumOfCirculations, maxNumOfNodes, verbose, keepBestSolution),
	_rand(rand), _chargeLevel(chargeLevel)
{
	assert(_rand && rand->getSize() >= _populationSize);
}


CuSolutionGeneratorAcoGpu::~CuSolutionGeneratorAcoGpu()
{
}


shared_ptr<CuPlans> CuSolutionGeneratorAcoGpu::run(shared_ptr<CuSelectionResult> selectionResult, float chanceOfRandomSelection)
{
	Stopwatch stopwatch;
	stopwatch.start();

	int blockSize; // Threads pro Block
	int minGridSize;
	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)antsRaceKernel));
	int numOfBlocks = (_populationSize + blockSize - 1) / blockSize;

	dim3 dimGrid(numOfBlocks, 1);
	dim3 dimBlock(blockSize, 1);

	int numOfEdges = Max_EdgesPerNode;	// TODO tatsächliche Höchstanzahl ermitteln, um Speicherplatz zu sparen.
	//int numOfEdges = _ptn->getHighestNumOfEdges();
	CuVector1<CuVector1<EdgeId>*> activeEdges(_populationSize);
	CuVector1<CuVector1<EdgeId>*> activeEdges_dev(_populationSize);
	CuVector1<CuVector1<float>*> weights(_populationSize);
	CuVector1<CuVector1<float>*> weights_dev(_populationSize);
	CuVector1<CuVector1<bool>*> isNodeActiveMat(_populationSize);
	CuVector1<CuVector1<bool>*> isNodeActiveMat_dev(_populationSize);
	for (int i = 0; i < _populationSize; i++) {
		isNodeActiveMat[i] = new CuVector1<bool>(_ptn->nodes.getNumOfNodes());
		isNodeActiveMat[i]->setAll(true);
		isNodeActiveMat_dev[i] = isNodeActiveMat[i]->getDevPtr();

		activeEdges[i] = new CuVector1<EdgeId>(numOfEdges);
		activeEdges_dev[i] = activeEdges[i]->getDevPtr();

		weights[i] = new CuVector1<float>(numOfEdges);
		weights_dev[i] = weights[i]->getDevPtr();
	}
	CuVector1<CuVector1<bool>*> *bv_dev = isNodeActiveMat_dev.getDevPtr();
	CuVector1<CuVector1<EdgeId>*> *ae_dev = activeEdges_dev.getDevPtr();
	CuVector1<CuVector1<float>*> *w_dev = weights_dev.getDevPtr();

	CuConstructionGraph *ptn_dev = _ptn->getDevPtr();
	CuProblem *problem_dev = _problem->getDevPtr();
	CuPlans *dp_dev = _plans->getDevPtr();
	RandomGpu *rand_dev = _rand->getDevPtr();
	ConsumptionMatrix *bc_dev = _batteryConsumption->getDevPtr();

	CuLockVector1 *nodeLock = new CuLockVector1(_ptn->nodes.getNumOfNodes());

	//CUDA_CHECK(cudaFuncSetCacheConfig(antsRaceKernel, cudaFuncCachePreferL1));

	antsRaceKernel << <dimGrid, dimBlock >> > (_populationSize, selectionResult->getDevPtr(), ptn_dev, problem_dev, dp_dev, rand_dev, 
		bv_dev, bc_dev, _chargeLevel, ae_dev, w_dev, nodeLock->getDevPtr(), _verbose, chanceOfRandomSelection, _keepBestSolution);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	delete nodeLock;

	CuVector1<float>* ptr;
	for (int i = 0; i < _populationSize; i++) {
		delete isNodeActiveMat[i];
		delete activeEdges[i];
		ptr = weights[i];
		delete ptr;
	}

	_plans->copyToHost();
	//_plans->dump(_ptn, 0);

	selectionResult->markAllAsAlive();
	selectionResult->copyToDevice();

	stopwatch.stop("Lösungen generiert (GPU): ");

	return _plans;
}

