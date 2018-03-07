#pragma once

#include "NeuralNetwork.h"

#include <Eigen/Dense>

#include <vector>

class NeuralNetworkTrainer
{

public:
	NeuralNetworkTrainer() = delete;
	NeuralNetworkTrainer(double lambda, double alpha, double tol, int maxIter);

public:
	void trainNeuralNetwork(NeuralNetwork& network, 
		const Eigen::VectorXd& input, const Eigen::VectorXd& output);

	double costFunction(const NeuralNetwork& network, 
		const Eigen::VectorXd& input, const Eigen::VectorXd& output);

	std::vector<Eigen::MatrixXd> backwardPropagate(const NeuralNetwork& network, 
		const Eigen::VectorXd& input);

	std::pair<int, double> gradientDescent(const NeuralNetwork& network, 
		const Eigen::VectorXd& input, const Eigen::VectorXd& output);

	void setLambda(double lambda);
	void setAlpha(double alpha);
	void setTolerance(double tol);
	void setMaxIter(double maxIter);

private:
	double lambda_ = 0;
	double alpha_ = 0;
	double tol_ = 0;
	int maxIter_ = 0;
};