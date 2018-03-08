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
	NeuralNetwork(const std::vector<SizeType>& layers);
	NeuralNetwork(const std::vector<Matrix>& weights);


public:

	Matrix forwardPropagate(const Matrix& input);

	const std::vector<Matrix>& getWeights();
	const std::vector<SizeType>& getLayers();

	/*Iterator insertLayer(Iterator pos);
	Iterator removeLayer(Iterator pos);
	void changeLayerSize(MatrixSize pos, const MatrixSize layerSize);
	MatrixSize getLayerSize(Iterator pos);*/

private:
	std::vector<Matrix> weights_;
	std::vector<SizeType> layers_;
};