#include "NeuralNetworkTrainer.h"

#include <algorithm>
#include <iostream>
#include <cassert>
#include <limits>

NeuralNetworkTrainer::NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter) noexcept
	: lambda_(lambda), alpha_(alpha), tol_(tol), maxIter_(maxIter)
{
}

std::pair<int, double> NeuralNetworkTrainer::trainNeuralNetwork(NeuralNetwork& network,
	Eigen::MatrixXd input, Eigen::MatrixXd output)
{
	return gradientDescent(network, input, output);
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

		if (!outputApprox.allFinite()) {
			std::cout << "Error: forward prop returned Inf values.\n";
		}
			
		if (outputApprox.hasNaN()) {
			std::cout << "Error: forward prop returned NaN values.\n";
		}
		if (outputApprox.maxCoeff() > 1) {
			std::cout << "Error: forward prop returned values above 1.\n";
		}
		if (outputApprox.minCoeff() == 0) {
			std::cout << "Error: forward prop returned a value that was zero.\n";
		}	
		if(outputApprox.minCoeff() < 0) {
			std::cout << "Error: forward prop returned negative values.\n";
		}

		static auto cwiseLog = [](double x) noexcept {return isinf(std::log(x)) ? -1e-20 : std::log(x); };

		// Create a filter with the output values that are zero

		auto temp = output.cwiseProduct(outputApprox.unaryExpr(cwiseLog))
					 + (1 - output.array()).matrix().cwiseProduct((1-outputApprox.array()).matrix().unaryExpr(cwiseLog));

		if (!temp.allFinite()) {
			std::cout << "Error: temp contains Inf values.\n";
		}

		costLog = -temp.rowwise().mean().sum();
	}
	
	// Cost attributed to regularization of weight matrices
	// This helps counter-act overfitting
	double costReg = 0; 
	for (const auto& w : network.weights_) {
		costReg += w.cwiseAbs2().sum() * lambda_ / (2 * NCOLS);
	}
	
	return costLog + costReg;
}

std::vector<Eigen::MatrixXd> NeuralNetworkTrainer::backwardPropagate(const NeuralNetwork& network,
	const Eigen::MatrixXd& input, const Eigen::MatrixXd& output)
{
	auto activations = forwardPropagateAll(network, input);

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
		const int nrows = i > 0 ? delta.rows() - 1 : delta.rows();
		
		jacobian[ri] = delta.block(0, 0, nrows, NCOLS) * a.transpose() / NCOLS;
		jacobian[ri].block(0, 0, w.rows(), w.cols() - 1) += lambda_ * w.block(0, 0, w.rows(), w.cols() - 1);

		// Calculate the error term for the next layer
		if (i < sz - 1) {
			delta = (w.transpose() * delta.block(0, 0, nrows, NCOLS)).cwiseProduct(a).cwiseProduct((a.array() - 1).matrix());
		}
	}

	return jacobian;
}

std::vector<Eigen::MatrixXd> NeuralNetworkTrainer::forwardPropagateAll(const NeuralNetwork& network,
	const Eigen::MatrixXd& input)
{
	static auto sigmoidFunc = [](double x) noexcept {return 1 / (1 + std::exp(-x)); };
	const int NCOLS = input.cols();

	Eigen::MatrixXd::Index nrowsMax = input.rows();
	for (const auto& w : network.weights_) {
		if (w.rows() > nrowsMax) {
			nrowsMax = w.rows();
		}
	}

	int nrows = input.rows();

	Eigen::MatrixXd activation(nrowsMax + 1, NCOLS);
	activation.block(0, 0, nrows, NCOLS) = input;
	activation.block(nrows, 0, 1, NCOLS).setOnes();

	std::vector<Eigen::MatrixXd> activations{ activation.block(0, 0, nrows + 1, NCOLS) };
	for (auto it = network.weights_.begin(); it != network.weights_.end(); ++it) {
		const auto& w = *it;
		activation.block(0, 0, w.rows(), NCOLS) = (w*activation.block(0, 0, nrows + 1, NCOLS)).unaryExpr(sigmoidFunc);
		
		nrows = w.rows();

		if (it != network.weights_.end() - 1) {
			activation.block(nrows, 0, 1, NCOLS).setOnes();
			activations.push_back(activation.block(0, 0, nrows + 1, NCOLS));
		}
		else {
			activations.push_back(activation.block(0, 0, nrows, NCOLS));
		}
	}

	return activations;
}

std::pair<int, double> NeuralNetworkTrainer::gradientDescent(NeuralNetwork& network,
	const Eigen::MatrixXd& input, const Eigen::MatrixXd& output)
{
	int i = 0;
	double prevCost = costFunction(network, input, output);
	double costDiff = 2 * tol_; // Initial value to ensure tol_ < stepSize for the first iteration
	for (; tol_ < costDiff && i < maxIter_; ++i) {
		std::cout << "Grad desc step " << i << " Curr cost: " << prevCost << std::endl;

		{
			const auto jacobian = backwardPropagate(network, input, output);
			if (jacobian.size() != network.weights_.size()) {
				std::cout << "Error: Jacobian does not agree with weight matrices.\n";
			}
			auto ita = network.weights_.begin();
			auto itb = jacobian.begin();
			while (ita != network.weights_.end() && itb != jacobian.end()) {
				*ita = *ita - alpha_ * *itb;
				++ita;
				++itb;
			}
		}

		double cost = costFunction(network, input, output);
		costDiff = prevCost - cost;
		prevCost = cost;

		if (costDiff < 0) {
			std::cout << "The cost increased. Stopping gradient descent.\n";
			++i;
			break;
		}
	}

	return { i, costDiff };
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> NeuralNetworkTrainer::normalizeFeatures(Eigen::MatrixXd& features)
{
	const auto NCOLS = features.cols();
	assert(NCOLS > 1 && "Cannot calculate standard deviation of one element vectors");

	auto ONES = Eigen::RowVectorXd::Ones(NCOLS);

	auto meanMat = features.rowwise().mean() * ONES;
	auto stdDevMat = (((features - meanMat) / std::sqrt(NCOLS - 1)).rowwise().norm() * ONES);
	// Think about how this affects later normalized forward propagation using new input data
	auto stdDevMatCorrected = stdDevMat.unaryExpr([](double v) { return std::abs(v) > 0 ? v : 1; });

	features = (features - meanMat).cwiseQuotient(stdDevMatCorrected);

	return { meanMat, stdDevMatCorrected };
}