#include "NeuralNetworkTrainer.h"

#include <algorithm>
#include <iostream>
#include <cassert>

NeuralNetworkTrainer::NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter) noexcept
	: lambda_(lambda), alpha_(alpha), tol_(tol), maxIter_(maxIter)
{
}

void NeuralNetworkTrainer::trainNeuralNetwork(NeuralNetwork& network,
	Eigen::MatrixXd input, Eigen::MatrixXd output)
{
	normalizeFeatures(input);
	for (auto& w : network.weights_) {
		w.setRandom();
	}
	const auto p = gradientDescent(network, input, output);
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
		static auto cwiseLog = [](double x) noexcept {return std::log(x); };

		Eigen::MatrixXd temp = output.cwiseProduct(outputApprox.unaryExpr(cwiseLog));
		temp += (ones - output).cwiseProduct((ones - outputApprox).unaryExpr(cwiseLog));

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

	// Jacobian of the cost function w.r.t each weight matrix
	std::vector<Eigen::MatrixXd> jacobian(sz);

	// Calculate the jacobian for each weight matrix, starting from the last.
	const auto NCOLS = input.cols(); // Number of training examples
	Eigen::MatrixXd delta = activations.back() - output; // End layer error term
	for (std::size_t i = 0; i < sz; ++i) {
		const auto ri = sz - 1 - i;
		const auto& a = activations[ri];
		const auto& w = network.weights_[ri];

		// Remove the bias unit portion of the error term. The end term has none
		if (i > 0) {
			delta.conservativeResize(delta.rows() - 1, Eigen::NoChange);
		}

		jacobian[ri] = delta * a.transpose() / NCOLS;
		jacobian[ri].block(0, 0, w.rows(), w.cols() - 1) += lambda_ * w.block(0, 0, w.rows(), w.cols() - 1);

		// Calculate the error term for the next layer
		delta = w.transpose() * delta; 
		delta = delta.cwiseProduct(a);
		delta = delta.cwiseProduct(Eigen::MatrixXd::Ones(a.rows(), a.cols()) - a);
	}

	return jacobian;
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

std::pair<int, double> NeuralNetworkTrainer::gradientDescent(NeuralNetwork& network,
	const Eigen::MatrixXd& input, const Eigen::MatrixXd& output)
{
	std::vector<Eigen::MatrixXd> prevWeights;
	prevWeights.reserve(network.weights_.size());

	int i = 0;
	double stepSize = 2 * tol_; // Initial value to ensure tol_ < stepSize for the first iteration
	for (; tol_ < stepSize && i < maxIter_; ++i) {
		auto& weights = network.weights_;
		prevWeights = weights;

		{
			const auto jacobian = backwardPropagate(network, input, output);
			auto ita = weights.begin();
			auto itb = jacobian.begin();
			while (ita != weights.end() && itb != jacobian.end()) {
				*ita = *ita - alpha_ * *itb;
				++ita;
				++itb;
			}
		}

		{
			std::vector<Eigen::MatrixXd> stepDiff;
			stepDiff.reserve(weights.size());
			auto ita = weights.begin();
			auto itb = prevWeights.begin();
			while (ita != weights.end() && itb != prevWeights.end()) {
				stepDiff.push_back((*ita - *itb).cwiseAbs());
				++ita;
				++itb;
			}

			auto it = std::max_element(stepDiff.begin(), stepDiff.end(), 
				[](const auto& m1, const auto& m2) {
				return m1.maxCoeff() < m1.maxCoeff();
			});

			stepSize = it->maxCoeff();
		}
	}

	return { i, stepSize };
}

void NeuralNetworkTrainer::normalizeFeatures(Eigen::MatrixXd& features)
{
	const auto NCOLS = features.cols();
	assert(NCOLS > 1 && "Cannot calculate standard deviation of one element vectors");

	for (int i = 0; i < features.rows(); ++i) {
		const double mean = features.row(i).sum() / NCOLS;
		const Eigen::RowVectorXd meanVec = mean * Eigen::RowVectorXd::Ones(NCOLS);

		const double stdDev = (features.row(i) - meanVec).norm() / (NCOLS - 1);

		features.row(i) = (features.row(i) - meanVec) / stdDev;
	}

	return;
}