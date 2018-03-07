#include "NeuralNetworkTrainer.h"

NeuralNetworkTrainer::NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter)
	: lambda_(lambda), alpha_(alpha), tol_(tol), maxIter_(maxIter)
{
}

NeuralNetworkTrainer& NeuralNetworkTrainer::setLambda(double lambda)
{
	lambda_ = lambda;
	return *this;
}

NeuralNetworkTrainer& NeuralNetworkTrainer::setAlpha(double alpha)
{
	alpha_ = alpha;
	return *this;
}

NeuralNetworkTrainer& NeuralNetworkTrainer::setTolerance(double tol)
{
	tol_ = tol;
	return *this;
}

NeuralNetworkTrainer& NeuralNetworkTrainer::setMaxIter(int maxIter)
{
	maxIter_ = maxIter;
	return *this;
}