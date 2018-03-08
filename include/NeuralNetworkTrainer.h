#pragma once

#include "NeuralNetwork.h"

#include <Eigen/Dense>

#include <vector>

class NeuralNetworkTrainer
{

	// The commented functions below are to be implemented.

public:
	NeuralNetworkTrainer() = default;
	NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter) noexcept;
	
public:
	// void trainNeuralNetwork(NeuralNetwork& network, 
	//	const Eigen::VectorXd& input, const Eigen::VectorXd& output);

	double costFunction(const NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output) const;

	// std::vector<Eigen::MatrixXd> backwardPropagate(const NeuralNetwork& network, 
	//	const Eigen::VectorXd& input);

	// std::pair<int, double> gradientDescent(const NeuralNetwork& network, 
	//	const Eigen::VectorXd& input, const Eigen::VectorXd& output);

	NeuralNetworkTrainer& setLambda(double lambda);
	NeuralNetworkTrainer& setAlpha(double alpha);
	NeuralNetworkTrainer& setTolerance(double tol);
	NeuralNetworkTrainer& setMaxIter(int maxIter);

private:
	double lambda_ = 0;
	double alpha_ = 0;
	double tol_ = 0;
	int maxIter_ = 0;
};