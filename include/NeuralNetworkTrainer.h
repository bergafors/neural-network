#pragma once

#include "NeuralNetwork.h"

#include <Eigen/Dense>

#include <vector>

struct NeuralNetworkTrainer
{
	double lambda_ = 0;
	double alpha_ = 0;
	double tol_ = 0;
	int maxIter_ = 0;

	// The commented functions below are to be implemented.

public:
	NeuralNetworkTrainer() = default;
	NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter) noexcept;
	
public:
	// void trainNeuralNetwork(NeuralNetwork& network, 
	//	const Eigen::VectorXd& input, const Eigen::VectorXd& output);

	double costFunction(const NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output) const;

	// Return the jacobian of the cost function w.r.t to the weight matrices
	// in %network, calculated given %input and %output
	std::vector<Eigen::MatrixXd> backwardPropagate(const NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output);

	// Return the unit activations of each network layer. Including the bias unit
	std::vector<Eigen::MatrixXd> forwardPropagateAll(const NeuralNetwork& network,
		const Eigen::MatrixXd& input);

	std::pair<int, double> gradientDescent(NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output);

};