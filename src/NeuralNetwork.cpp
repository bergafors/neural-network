#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<SizeType>& layers)
	: layers_(layers)
{
	// Add 1 to the column size to account for the bias unit

	if (layers_.size() >= 2) {
		weights_.reserve(layers_.size() - 1);
		for (auto it = layers_.begin(); it != layers_.end() - 1; ++it) {
			weights_.push_back(Matrix(*(it + 1), 1 + *it));
		}
	}
	else {
		throw std::logic_error("Number of layers has to be >= 2.");
	}
}

NeuralNetwork::NeuralNetwork(const std::vector<Matrix>& weights)
	: weights_(weights)
{
	layers_.push_back(weights_.front().cols() - 1);
	for (const auto& w : weights_) {
		layers_.push_back(w.rows());
	}
}

NeuralNetwork::Matrix NeuralNetwork::forwardPropagate(const Matrix& input) const
{
	static auto sigmoidFunc = [](double x) noexcept {return 1 / (1 + std::exp(-x)); };

	const int NCOLS = input.cols();

	Eigen::MatrixXd::Index nrowsMax = input.rows();
	for (const auto& w : weights_) {
		if (w.rows() > nrowsMax) {
			nrowsMax = w.rows();
		}
	}

	Eigen::MatrixXd activation(nrowsMax + 1, NCOLS);
	activation.block(0, 0, input.rows(), NCOLS) = input;

	int nrows = input.rows();
	for (const auto& w : weights_) {
		activation.block(nrows, 0, 1, NCOLS).setOnes();

		activation.block(0, 0, w.rows(), NCOLS) = (w*activation.block(0, 0, nrows + 1, NCOLS)).unaryExpr(sigmoidFunc);
		nrows = w.rows();
	}

	return activation.block(0, 0, nrows, NCOLS);
}

std::vector<NeuralNetwork::Matrix>& NeuralNetwork::getWeights() noexcept
{
	return weights_;
}

const std::vector<NeuralNetwork::Matrix>& NeuralNetwork::getWeights() const noexcept
{
	return weights_;
}
const std::vector<NeuralNetwork::SizeType>& NeuralNetwork::getLayers() const noexcept
{
	return layers_;
}