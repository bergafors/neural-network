#include "NeuralNetwork.h"
#include <iterator>


NeuralNetwork::NeuralNetwork(std::vector<SizeType> layers)
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
		throw std::logic_error("Size of layerSizes has to be >= 2.");
	}
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