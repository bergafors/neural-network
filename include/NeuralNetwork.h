#pragma once

#include "NeuralNetworkTrainer.h"

#include <Eigen/Dense>

#include <vector>

class NeuralNetwork
{
public:
	friend class NeuralNetworkTrainer;

	using Matrix = Eigen::MatrixXd;
	using SizeType = Matrix::Index;

	// The commented functions below are to be implemented.

public:
	NeuralNetwork() = delete;
	//NeuralNetwork(MatrixSize inputLayerSize, MatrixSize outputLayerSize);
	NeuralNetwork(std::vector<SizeType> layerSizes);

public:

	Matrix forwardPropagate(const Matrix& input);

	/*Iterator insertLayer(Iterator pos);
	Iterator removeLayer(Iterator pos);
	void changeLayerSize(MatrixSize pos, const MatrixSize layerSize);
	MatrixSize getLayerSize(Iterator pos);*/

private:
	std::vector<Matrix> weights_;
	std::vector<SizeType> layers_;
};