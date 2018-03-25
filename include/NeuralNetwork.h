#pragma once

#include <Eigen/Dense>
#include <vector>

class NeuralNetwork
{
public:
	friend struct NeuralNetworkTrainer;

public:
	NeuralNetwork() = delete;
	// Create a neural network with layer structure given by %layers.
	// Note that the values in %layers should not include the bias unit.
	// Requirement: layers.size() >= 2.
	NeuralNetwork(const std::vector<Eigen::MatrixXd::Index>& layers);
	// Create a neural network with layer structure given by %weights,
	// the corresponding weight matrices.
	NeuralNetwork(const std::vector<Eigen::MatrixXd>& weights);


public:
	// Return the output layer activation for a given %input
	Eigen::MatrixXd forwardPropagate(const Eigen::MatrixXd& input) const;

	std::vector<Eigen::MatrixXd>& getWeights() noexcept;
	const std::vector<Eigen::MatrixXd>& getWeights() const noexcept;
	const std::vector<Eigen::MatrixXd::Index>& getLayers() const noexcept;

private:
	std::vector<Eigen::MatrixXd> weights_;
	std::vector<Eigen::MatrixXd::Index> layers_;
};