#include "NeuralNetworkTrainer.h"

NeuralNetworkTrainer::NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter) noexcept
	: lambda_(lambda), alpha_(alpha), tol_(tol), maxIter_(maxIter)
{
}

double NeuralNetworkTrainer::costFunction(const NeuralNetwork& network,
	const Eigen::MatrixXd& input, const Eigen::MatrixXd& output) const
{
	const auto NROWS = output.rows(); // Output size
	const auto NCOLS = output.cols(); // Number of training examples
	
	const auto outputApprox = network.forwardPropagate(input);

	// Cost attributed to logistic regression
	double costLog = 0;

	const auto ones = NeuralNetwork::Matrix::Ones(NROWS, NCOLS);
	auto temp = output.cwiseProduct(outputApprox) + (ones - output).cwiseProduct(ones - outputApprox);
	
	costLog = -temp.sum() / NCOLS;

	// Cost attributed to regularization of weight matrices
	// This counter-acts overfitting
	double costReg = 0; 
	for (const auto& w : network.weights_) {
		costReg += w.cwiseAbs2().sum();
	}
	costReg *= lambda_ / (2 * NCOLS);
	
	return costLog + costReg;
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