#pragma once

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

#include <fstream>
#include <string>

#include <vector>
#include <iterator>

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
	// Check that backprop returns matrices of the same
	// dimensions as the network weight matrices

	NeuralNetwork::Matrix w1(2, 3);
	NeuralNetwork::Matrix w2(1, 3);
	NeuralNetwork nn({ w1, w2 });

	NeuralNetwork::Matrix input(2, 4);
	NeuralNetwork::Matrix output(1, 4);
	input << 0, 0, 1, 1, 0, 1, 0, 1;
	output << 1, 0, 0, 1;

	NeuralNetworkTrainer nnt;

	REQUIRE_NOTHROW(nnt.backwardPropagate(nn, input, output));

	const auto jacobian = nnt.backwardPropagate(nn, input, output);

	auto sameDim = [](const auto& mveca, const auto& mvecb) {
		auto ita = mveca.begin();
		auto itb = mvecb.begin();
		bool same = true;
		while (ita != mveca.end() && itb != mvecb.end()) {
			if (ita->rows() != itb->rows() || ita->cols() != itb->cols()) {
				std::cout << "Jacobian and weight matrix number " 
					<< std::distance(mveca.begin(), ita) << " differ.\n";
				std::cout << "Jacobian dim: " << "(" << ita->rows() << ", " << ita->cols() << ")\n";
				std::cout << "Weight matrix dim: " << "(" << itb->rows() << ", " << itb->cols() << ")\n";

				same = false;
			}

			++ita;
			++itb;
		}

		return same;
	};

	REQUIRE(sameDim(jacobian, nn.getWeights()));
}

TEST_CASE("Backward propagation II")
{
	// Check that the values returned by back prop agree with 
	// the values computed using the Newton quotient

	NeuralNetwork::Matrix w1(2, 3);
	NeuralNetwork::Matrix w2(1, 3);
	NeuralNetwork nn({ w1.setRandom(), w2.setRandom() });

	NeuralNetwork::Matrix input(2, 4);
	NeuralNetwork::Matrix output(1, 4);
	input << 0, 0, 1, 1, 0, 1, 0, 1;
	output << 1, 0, 0, 1;

	NeuralNetworkTrainer nnt;

	auto jacobian = nnt.backwardPropagate(nn, input, output);

	const double EPS = 1e-4;
	std::vector<Eigen::MatrixXd> newtonQuot;
	auto nnMinus = nn;
	auto nnPlus = nn;
	for (std::size_t k = 0; k < nn.getWeights().size(); ++k) {
		const auto& w = nn.getWeights()[k];
		Eigen::MatrixXd nq(w.rows(), w.cols());
		for (int j = 0; j < w.cols(); ++j) {
			for (int i = 0; i < w.rows(); ++i) {
				nnMinus.getWeights()[k](i, j) -= EPS;
				nnPlus.getWeights()[k](i, j) += EPS;
				nq(i, j) = (nnt.costFunction(nnPlus, input, output) - nnt.costFunction(nnMinus, input, output)) / (2 * EPS);
				nnMinus.getWeights()[k](i, j) += EPS;
				nnPlus.getWeights()[k](i, j) -= EPS;
			}
		}
		newtonQuot.push_back(std::move(nq));
	}

	auto agrees = [](const auto& mveca, const auto& mvecb) {
		for (std::size_t k = 0; k < mveca.size(); ++k) {
			const auto maxRelErr = (mveca[k] - mvecb[k]).cwiseQuotient(mveca[k]).cwiseAbs().maxCoeff();
			if ( maxRelErr > TOL) {
				std::cout << "A maximum relative error of " << maxRelErr << " was detected.\n";
				std::cout << "This exceeds the tolerance level of " << TOL << '\n';
				return false;
			}
		}
		return true;
	};

	REQUIRE(agrees(jacobian, newtonQuot));
}

TEST_CASE("Gradient descent")
{
	NeuralNetwork::Matrix w1(2, 3);
	NeuralNetwork::Matrix w2(1, 3);
	NeuralNetwork nn({ w1.setRandom(), w2.setRandom() });

	NeuralNetwork::Matrix input(2, 4);
	NeuralNetwork::Matrix output(1, 4);
	input << 0, 0, 1, 1, 0, 1, 0, 1;
	output << 1, 0, 0, 1;

	NeuralNetworkTrainer nnt(0, 1, 1, 50);
	
	REQUIRE_NOTHROW(nnt.gradientDescent(nn, input, output));
}

TEST_CASE("Gradient descent II")
{
	NeuralNetwork::Matrix w1(2, 3);
	w1 << 20, 20, -30, -20, -20, 10;

	NeuralNetwork::Matrix w2(1, 3);
	w2 << 20, 20, -10;

	NeuralNetwork nn({ w1, w2});

	NeuralNetwork::Matrix input(2, 4);
	NeuralNetwork::Matrix output(1, 4);
	input << 0, 0, 1, 1, 0, 1, 0, 1;
	output << 1, 0, 0, 1;

	NeuralNetworkTrainer nnt(0, 1e-2, TOL/1000, 1000);

	auto p = nnt.gradientDescent(nn, input, output);
	//std::cout << p.first << " " << p.second;
	
	REQUIRE(nnt.costFunction(nn, input, output) < TOL*10);
}

TEST_CASE("Normalize features")
{
	Eigen::MatrixXd someMat(10, 8);
	someMat.setRandom();

	NeuralNetworkTrainer nnt;

	REQUIRE_NOTHROW(nnt.normalizeFeatures(someMat));
}

TEST_CASE("Large training set")
{
	std::ifstream trainData("../testdata/train-images.idx3-ubyte", std::ios::binary);
	if (!trainData) {
		std::cout << "Couldnt open data file";
	}

	long nread = 0;
	try {
		trainData.exceptions(std::ifstream::badbit | std::ifstream::failbit);
		std::int32_t mgn = 0;
		trainData.read((char*) &mgn, 4);
		std::cout << std::hex << mgn << std::dec << std::endl;
		unsigned char pixel;
		trainData.ignore(12);
		int i = 0;
		while (true) {
			if (i == (28 * 28)) {
				++nread;
				if (nread % 1000 == 0)
					std::cout << nread << std::endl;
				i = 0;
			}
			trainData.read((char*)&pixel, sizeof(pixel));
			++i;
		}
	}
	catch (std::ifstream::failure&) {
		std::cout << "Error reading file\n";
	}

	std::cout << nread << std::endl;
}