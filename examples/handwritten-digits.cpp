/*
This example trains a neural network with one hidden layer to detect images of handwritten digits.
The required training data can be found at http://yann.lecun.com/exdb/mnist/.

The example should be compiled with -O2 and -mssd2. To use the whole training set (60000 examples)
it is necessary to compile it on a 64-bit system.

Note that the some entries in the training data uses the big-endian format. readData assumes
that the system this is compiled against uses small-endian. It transforms the entries to
small-endian using _byteswap_ulong from intrin.h, an MSVC-only library. 
*/


#pragma once

#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

#include "Eigen\Dense"

#include <iostream>
#include <fstream>
#include <vector>

#include <cmath>
#include <ctime>
#include <intrin.h>

enum class DataType {Images, Labels};

Eigen::MatrixXd readData(std::ifstream& file, const DataType dt, const std::int32_t maxItems = -1);

int main()
{
	// Create a trainer with regularization parameter lambda_ = 0.01;
	// training rate alpha_ = 0.03; cost function difference tolerance tol_ = 1e-3;
	// and maximum number of iterations maxIter_ = 10. The trainer will use mini-batch
	// gradient descent.
	NeuralNetworkTrainer nnt(0.01, 0.03, 1e-6, 10, NeuralNetworkTrainer::GradientDescentType::MINIBATCH);
	NeuralNetwork nn({ 28 * 28, 30, 10 });

	// Initialize weights randomly using Xavier-inititalization
	std::srand((unsigned int)time(0));
	for (auto& w : nn.getWeights()) {
		w.setRandom();
		const double eInit = std::sqrt(2.0 / (w.cols() + w.rows()));
		w *= eInit;
	}

	// Going above 40000 may cause issues on 32-bit systems
	const std::int32_t maxTrainItems = 40000;
	const std::int32_t maxTestItems = 10000;

	Eigen::MatrixXd trainInput, trainOutput, testInput, testOutput;

	{
		std::ifstream file("../testdata/train-images.idx3-ubyte", std::ios::binary);
		if (!file) {
			std::cout << "Couldnt open data file";
		}
		trainInput = readData(file, DataType::Images, maxTrainItems);
		std::cout << '\r' << "Training input data read.\n";
	}

	{
		std::ifstream file("../testdata/t10k-images.idx3-ubyte", std::ios::binary);
		if (!file) {
			std::cout << "Couldnt open data file";
		}
		testInput = readData(file, DataType::Images, maxTestItems);
		std::cout << '\r' << "Test input data read.\n";
	}

	{
		std::ifstream file("../testdata/train-labels.idx1-ubyte", std::ios::binary);
		if (!file) {
			std::cout << "Couldnt open data file";
		}
		trainOutput = readData(file, DataType::Labels, maxTrainItems);
		std::cout << '\r' << "Training output data read.\n";
	}

	{
		std::ifstream file("../testdata/t10k-labels.idx1-ubyte", std::ios::binary);
		if (!file) {
			std::cout << "Couldnt open data file";
		}
		testOutput = readData(file, DataType::Labels, maxTestItems);
		std::cout << '\r' << "Test output data read.\n";
	}

	std::cout << "Normalizing training and test input features...\n";
	nnt.normalizeFeatures(trainInput);
	nnt.normalizeFeatures(testInput);
	std::cout << "Normalization complete.\n";

	if (trainInput.hasNaN()) {
		std::cout << "Error: normalized training input contains NaN values\n";
	}
	if (!trainInput.allFinite()) {
		std::cout << "Error: normalized training input contains Inf values\n";
	}

	if (testInput.hasNaN()) {
		std::cout << "Error: normalized test input contains NaN values\n";
	}
	if (!testInput.allFinite()) {
		std::cout << "Error: normalized test input contains Inf values\n";
	}

	std::cout << "Training neural network...\n";
	nnt.trainNetwork(nn, trainInput, trainOutput, testInput, testOutput);
	std::cout << "Training complete.\n";

	std::cout << "Calculating accuracy...\n";
	double accuracy = nnt.predict(nn, testInput, testOutput);
	accuracy /= testInput.cols();

	std::cout << "The neural network correctly identified " << 100*accuracy << "% of the hand-written digits.\n";

	return 0;
}

Eigen::MatrixXd readData(std::ifstream& file, const DataType dt, const std::int32_t maxItems)
{
	Eigen::MatrixXd data;

	try {
		file.exceptions(std::ifstream::badbit | std::ifstream::failbit);

		std::int32_t magicNum = 0;
		file.read((char*)&magicNum, sizeof(magicNum));
		// Stored in big-endian format. Intel processors use little-endian
		magicNum = _byteswap_ulong(magicNum); 

		std::int32_t nitems = 0;
		file.read((char*)&nitems, sizeof(nitems));
		nitems = _byteswap_ulong(nitems);

		std::int32_t nrows = 0, ncols = 0;
		if (dt == DataType::Images) {
			file.read((char*)&nrows, sizeof(nrows));
			file.read((char*)&ncols, sizeof(ncols));
			nrows = _byteswap_ulong(nrows);
			ncols = _byteswap_ulong(ncols);
		}
		else /*if (dt == DataType::Labels) */ {
			nrows = 10;
			ncols = 1;
		}

		if (maxItems > 0 && nitems > maxItems) {
			nitems = maxItems;
		}
		
		data.resize(nrows*ncols, nitems);
		data.setZero();

		unsigned char element;
		for (int k = 0; k < nitems; ++k) {
			for (int i = 0; i < nrows; ++i) {
				for (int j = 0; j < ncols; ++j) {
					file.read((char*)&element, sizeof(element));
					if (dt == DataType::Images) {
						data(i*ncols + j, k) = element;
					}
					else /*if (dt == DataType::Labels) */ {
						data(element, k) = 1;
						i = nrows;
					}
				}
			}
			if (nitems < 100 || (k + 1) % (nitems / 100) == 0) {
				std::cout << '\r' << ((k + 1)*100) / nitems << "% of items read." << std::flush;
			}
		}
	}
	catch (std::ifstream::failure&) {
		std::cout << "Error reading file\n";
	}

	return data;
}

