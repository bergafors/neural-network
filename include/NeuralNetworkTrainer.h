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
	 std::pair<int, double> trainNeuralNetwork(NeuralNetwork& network, 
		Eigen::MatrixXd input, Eigen::MatrixXd output);

	double costFunction(const NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output) const;

	// Return the jacobian of the cost function w.r.t to the weight matrices
	// in %network, calculated given %input and %output
	std::vector<Eigen::MatrixXd> backwardPropagate(const NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output);

	// Return the unit activations of each network layer.
	std::vector<Eigen::MatrixXd> forwardPropagateAll(const NeuralNetwork& network,
		const Eigen::MatrixXd& input);

	// Perform gradient descent on %network and return the final
	// number of steps taken, and the final reduction of the cost function.
	std::pair<int, double> gradientDescent(NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output);

	// Normalize each row and return the matrices containing the mean and standard deviation,
	// respectively, of the initial values of each row.
	std::pair<Eigen::MatrixXd, Eigen::MatrixXd> normalizeFeatures(Eigen::MatrixXd& features);

};