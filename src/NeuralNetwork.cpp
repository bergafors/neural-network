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
	static auto sigmoidFunction = [](double x) noexcept {return 1 / (1 + std::exp(-x)); };

	// The unit activation of the input layer is just the input
	auto activation = input;
	for (const auto& m : weights_) {
		// Add bias unit to the activation
		activation.conservativeResize(activation.rows() + 1, Eigen::NoChange);
		activation.row(activation.rows() - 1).setOnes();

		// Calculate the unit activation in the next layer
		activation = (m*activation).unaryExpr(sigmoidFunction);
	}

	return activation;
}

const std::vector<NeuralNetwork::Matrix>& NeuralNetwork::getWeights() const noexcept
{
	return weights_;
}
const std::vector<NeuralNetwork::SizeType>& NeuralNetwork::getLayers() const noexcept
{
	return layers_;
}

/*NeuralNetwork::Iterator NeuralNetwork::insertLayer(Iterator pos, const MatrixSize layerSize)
{
	// Add 1 to the column size to account for the bias unit

	if (weights_.empty()) {
		layerSizes_.push_back(layerSize);
		return pos;
	}
	else if (pos == weights_.end()) {
		weights_.push_back(Matrix(layerSize, layerSizes_.back() + 1));
		layerSizes_.push_back(layerSize);
		return ++pos;
	}
	else {
		pos->resize(Eigen::NoChange, layerSize + 1);
		(pos - 1)->resize(layerSize + 1, Eigen::NoChange);

		const auto dist = std::distance(weights_.begin(), pos);
		auto posLaye = layerSizes_.begin() + 1 + dist; // Add 1 to rea
		pos = weights_.insert(pos, Matrix(layerSize, *(it - 1) + 1);

	}
}*/