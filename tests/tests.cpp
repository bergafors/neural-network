#pragma once

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

#include <vector>

#define TOL 1e-4 // The tolerence for various measures used in the tests

TEST_CASE("NeuralNetwork basic functionalities")
{
	NeuralNetwork nn{ std::vector<NeuralNetwork::SizeType>{1, 2} };
}

TEST_CASE("NeuralNetworkTrainer basic functionalities")
{
	NeuralNetworkTrainer nnt(0, 0, 0, 0);
}

TEST_CASE("Forward propagation I")
{
	// Check that forward propagation returns a matrix of the appropiate size.
	// Also check that forward prop returns a zero matrix if all weight 
	// matrices are zero matrices

	// Network with some layer configuration
	NeuralNetwork nn({ 9, 8, 7, 6, 5, 1, 2, 3, 4 });

	const auto nInput = nn.getLayers().front();
	const auto nOutput = nn.getLayers().back();

	// Number of training examples
	const NeuralNetwork::SizeType nExamples = 10;

	NeuralNetwork::Matrix input(nInput, nExamples);
	input.setRandom();

	auto output = nn.forwardPropagate(input);
	
	REQUIRE(output.rows() == nOutput);
	REQUIRE(output.cols() == nExamples);
	REQUIRE(output.sum() == 0);

}

TEST_CASE("Forward propagation II")
{
	// Emulate an XNOR-gate using a neural network
	// Signature: [bool; bool] -> bool

	// Emulate a AND b (first row) and a NOR b (second row)
	NeuralNetwork::Matrix w1(2, 3);
	w1 << 20, 20, -30, -20, -20, 10;

	// Emulate OR
	NeuralNetwork::Matrix w2(1, 3);
	w2 << 20, 20, -10;

	NeuralNetwork nn({ w1, w2 });

	// Create a truth table for XNOR
	NeuralNetwork::Matrix input(2, 4);
	NeuralNetwork::Matrix outputExact(1, 4);
	input << 0, 0, 1, 1, 0, 1, 0, 1;
	outputExact << 1, 0, 0, 1;

	auto output = nn.forwardPropagate(input);
	auto mean = (output - outputExact).cwiseAbs().mean();

	REQUIRE(std::abs(mean) < TOL);
}

TEST_CASE("Cost function I")
{
	// Emulate an XNOR-gate as in Forward propagation II.
	// Check that the cost function has a local minimum
	// for the supplied neural network

	// Note that this may test may fail, as the local minimum
	// is more like a saddle point in this case.

	// Num of networks with perturbed weight matrices to compare with
	const int NITER = 10; 

	NeuralNetwork::Matrix w1(2, 3);
	w1 << 20, 20, -30, -20, -20, 10;

	NeuralNetwork::Matrix w2(1, 3);
	w2 << 20, 20, -10;

	NeuralNetwork nn({ w1, w2 });

	NeuralNetwork::Matrix input(2, 4);
	NeuralNetwork::Matrix output(1, 4);
	input << 0, 0, 1, 1, 0, 1, 0, 1;
	output << 1, 0, 0, 1;

	NeuralNetworkTrainer nnt;

	const auto minCost = nnt.costFunction(nn, input, output);
	bool isMin = true;

	// Generate perturbations of the minimal network above and compare their cost to minCost
	std::vector<NeuralNetwork::Matrix> perturbedWeights;
	perturbedWeights.reserve(nn.getWeights().size());
	for (int i = 0; i < NITER; ++i) {
		for (const auto& w : nn.getWeights()) {
			auto perturbation = NeuralNetwork::Matrix::Random(w.rows(), w.cols()).cwiseProduct(w.cwiseAbs());
			auto pwm = w + perturbation;
			perturbedWeights.push_back(pwm);
		}

		NeuralNetwork nnPert(perturbedWeights);
		auto cost = nnt.costFunction(nnPert, input, output);

		if (cost < minCost) {
			isMin = false;
			break;
		}

		perturbedWeights.clear();
	}

	REQUIRE(isMin);
}

TEST_CASE("Backward propagation I")
{
	// Basic test to check if backprop throws an error

	NeuralNetwork::Matrix w1(2, 3);
	NeuralNetwork::Matrix w2(1, 3);
	NeuralNetwork nn({ w1, w2 });

	NeuralNetwork::Matrix input(2, 4);
	NeuralNetwork::Matrix output(1, 4);
	input << 0, 0, 1, 1, 0, 1, 0, 1;
	output << 1, 0, 0, 1;

	NeuralNetworkTrainer nnt;

	REQUIRE_NOTHROW(nnt.backwardPropagate(nn, input, output));
}