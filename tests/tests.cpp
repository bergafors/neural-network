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

TEST_CASE("Forward propagation")
{
	// Emulate an XNOR-gate using a neural network
	// Signature: [bool; bool] -> bool

	// Emulate a AND b (first row) and a NOR b (second row)
	NeuralNetwork::Matrix w1(2, 3);
	w1 << 20, 20, -30, -20, -20, 10;

	// Emulate OR
	NeuralNetwork::Matrix w2(1, 3);
	w2 << 20, 20, -10;

	NeuralNetwork nn({w1, w2});

	// Create a truth table for XNOR
	NeuralNetwork::Matrix input(2, 4);
	NeuralNetwork::Matrix correctOutput(1, 4);

	input << 0, 0, 1, 1, 0, 1, 0, 1;
	correctOutput << 1, 0, 0, 1;

	auto output = nn.forwardPropagate(input);
	auto norm = (output - correctOutput).norm();

	REQUIRE(norm < TOL);
}

TEST_CASE("NeuralNetworkTrainer basic functionalities")
{
	NeuralNetworkTrainer nnt(0, 0, 0, 0);
}