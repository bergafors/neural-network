#pragma once

#include <Eigen/Dense>

#include <vector>

class NeuralNetwork
{
public:
	friend struct NeuralNetworkTrainer;

	using Matrix = Eigen::MatrixXd;
	using SizeType = Matrix::Index;

public:
	NeuralNetwork() = delete;
	NeuralNetwork(const std::vector<SizeType>& layers);
	NeuralNetwork(const std::vector<Matrix>& weights);


public:
	Matrix forwardPropagate(const Matrix& input) const;

	std::vector<Matrix>& getWeights() noexcept;
	const std::vector<Matrix>& getWeights() const noexcept;
	const std::vector<SizeType>& getLayers() const noexcept;

private:
	std::vector<Matrix> weights_;
	std::vector<SizeType> layers_;
};