#pragma once

#include "NeuralNetworkTrainer.h"

#include <Eigen/Dense>

#include <vector>


class NeuralNetwork
{
	friend NeuralNetworkTrainer;

public:
	NeuralNetwork() = default;
	NeuralNetwork(std::vector<Eigen::MatrixXd::Index> layerSizes);

public:
	Eigen::VectorXd forwardPropagate(const Eigen::VectorXd& input);

	void addLayer(Eigen::MatrixXd::Index layerSize);
	Eigen::MatrixXd::Index getLayerSize(std::size_t index);

private:
	std::vector<Eigen::MatrixXd> weights_;
	std::vector<Eigen::MatrixXd::Index> layerSizes_;
};