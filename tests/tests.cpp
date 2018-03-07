#pragma once

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

#include <vector>

TEST_CASE("NeuralNetwork basic functionalities")
{
	NeuralNetwork nn{ std::vector<NeuralNetwork::SizeType>{1, 2} };
}

TEST_CASE("NeuralNetworkTrainer basic functionalities")
{
	NeuralNetworkTrainer nnt(0, 0, 0, 0);
}