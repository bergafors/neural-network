#include "NeuralNetworkTrainer.h"
#include <iostream>

NeuralNetworkTrainer::NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter) noexcept
	: lambda_(lambda), alpha_(alpha), tol_(tol), maxIter_(maxIter)
{
}

double NeuralNetworkTrainer::costFunction(const NeuralNetwork& network,
	const Eigen::MatrixXd& input, const Eigen::MatrixXd& output) const
{
	const auto NROWS = output.rows(); // Output size
	const auto NCOLS = output.cols(); // Number of training examples
	
	// Cost attributed to logistic regression								  
	double costLog = 0;

	{
		const auto outputApprox = network.forwardPropagate(input);

		const auto ones = NeuralNetwork::Matrix::Ones(NROWS, NCOLS);
		const auto temp = output.cwiseProduct(outputApprox) + (ones - output).cwiseProduct(ones - outputApprox);

		costLog = -temp.sum() / NCOLS;
	}
	
	// Cost attributed to regularization of weight matrices
	// This helps counter-act overfitting
	double costReg = 0; 
	for (const auto& w : network.weights_) {
		costReg += w.cwiseAbs2().sum();
	}
	costReg *= lambda_ / (2 * NCOLS);
	
	return costLog + costReg;
}

std::vector<Eigen::MatrixXd> NeuralNetworkTrainer::backwardPropagate(const NeuralNetwork& network,
	const Eigen::MatrixXd& input, const Eigen::MatrixXd& output)
{
	auto activations = forwardPropagateAll(network, input);

	// Add bias unit to each layer activation except the last one
	for (auto it = activations.begin(); it != activations.end() - 1; ++it) {
		it->conservativeResize(it->rows() + 1, Eigen::NoChange);
		it->row(it->rows() - 1).setOnes();
	}

	const auto sz = network.getWeights().size();

	// Partial derivates of the cost function
	std::vector<Eigen::MatrixXd> pdCost;
	pdCost.reserve(sz);

	const auto NCOLS = input.cols(); // Number of training examples
	Eigen::MatrixXd delta = activations.back() - output; // End layer error term
	for (std::size_t i = 0; i < sz; ++i) {
		const auto ri = sz - 1 - i;
		const auto& a = activations[ri];
		const auto& w = network.getWeights()[ri];

		pdCost.push_back(delta * a.transpose() / NCOLS);
		if (i < sz - 1) 
			pdCost.back() += lambda_ * w;

		if (i > 0) 
			delta.conservativeResize(delta.rows() - 1, Eigen::NoChange);

		// Calculate the error term for the next layer
		delta = w.transpose() * delta; 
		delta = delta.cwiseProduct(a);
		delta = delta.cwiseProduct(Eigen::MatrixXd::Ones(a.rows(), a.cols()) - a);
	}

	return pdCost;
}

std::vector<Eigen::MatrixXd> NeuralNetworkTrainer::forwardPropagateAll(const NeuralNetwork& network,
	const Eigen::MatrixXd& input)
{
	static auto sigmoidFunction = [](double x) noexcept {return 1 / (1 + std::exp(-x)); };

	// The unit activation of the input layer is just the input
	std::vector<Eigen::MatrixXd> activations{ input };
	for (const auto& w : network.weights_) {
		auto a = activations.back();

		// Add bias unit to the activation
		a.conservativeResize(a.rows() + 1, Eigen::NoChange);
		a.row(a.rows() - 1).setOnes();

		// Calculate the unit activation in the next layer
		a = (w*a).unaryExpr(sigmoidFunction);

		activations.push_back(std::move(a));
	}

	return activations;
}