#pragma once

#include "NeuralNetwork.h"
#include <Eigen/Dense>
#include <vector>

struct NeuralNetworkTrainer
{
	// Set gradient descent type
	// BATCH: regular batch gradient descent
	// MINIBATCH: stochastic gradient descent with mini-batch size 10
	enum class GradientDescentType { BATCH, MINIBATCH};

	double lambda_ = 0;
	double alpha_ = 0;
	double tol_ = 0;
	int maxIter_ = 0;
	GradientDescentType gdt_ = GradientDescentType::BATCH;

public:
	NeuralNetworkTrainer() = default;
	NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter, 
		GradientDescentType gdt = GradientDescentType::BATCH) noexcept;
	
public:
	// Train %network on given %input and %output training data.
	// Optionally provide test data to track the test cost.
	// Uses batch or mini-batch gradient descent as specified by
	// %gdt_.
	// Returns the number of iterations performed, as well
	// as the final training cost function difference when
	// using batch gradient descent.
	std::pair<int, double> trainNetwork(NeuralNetwork& network,
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output,
		const Eigen::MatrixXd& testInput = Eigen::MatrixXd{}, 
		const Eigen::MatrixXd& testOutput = Eigen::MatrixXd{});

	double costFunction(const NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output) const;

	// Return the jacobian of the cost function w.r.t to the weight matrices
	// in %network, calculated given %input and %output
	std::vector<Eigen::MatrixXd> backwardPropagate(const NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output);

	// Return the unit activations of each network layer.
	std::vector<Eigen::MatrixXd> forwardPropagateAll(const NeuralNetwork& network,
		const Eigen::MatrixXd& input);

	// Perform batch gradient descent on %network and return the final
	// number of steps taken, and the final reduction of the cost function.
	std::pair<int, double> gradientDescent(NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output,
		const Eigen::MatrixXd& testInput, const Eigen::MatrixXd& testOutput);

	// Perform mini-batch gradient descent on %network
	std::pair<int, double> stochasticGradientDescent(NeuralNetwork& network,
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output,
		const Eigen::MatrixXd& testInput, const Eigen::MatrixXd& testOutput);

	// Normalize %features over each row
	void normalizeFeatures(Eigen::MatrixXd& features);

	// Calculate how many examples in &output that &network correctly
	// predicts given %input. Return the number of accurately
	// predicted examples.
	Eigen::MatrixXd::Index predict(const NeuralNetwork& network, 
		const Eigen::MatrixXd& input, const Eigen::MatrixXd& output);

};