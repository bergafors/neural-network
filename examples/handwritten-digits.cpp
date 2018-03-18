#pragma once

#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

#include "Eigen\Dense"

#include <iostream>
#include <fstream>
#include <vector>

#include <cmath>
#include <intrin.h>

enum class DataType {Images, Labels};

Eigen::MatrixXd readData(std::ifstream& file, const DataType dt, const std::int32_t maxItems = -1);
double calculateAccuracy(const NeuralNetwork& nn, const Eigen::MatrixXd& input,
	const Eigen::MatrixXd& correctOutput);

int main()
{
	const std::int32_t maxTrainItems = 50;
	const std::int32_t maxTestItems = 10;

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
		std::ifstream file("../testdata/train-labels.idx1-ubyte", std::ios::binary);
		if (!file) {
			std::cout << "Couldnt open data file";
		}
		trainOutput = readData(file, DataType::Labels, maxTrainItems);
		std::cout << '\r' << "Training output data read.\n";
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
		std::ifstream file("../testdata/t10k-labels.idx1-ubyte", std::ios::binary);
		if (!file) {
			std::cout << "Couldnt open data file";
		}
		testOutput = readData(file, DataType::Labels, maxTestItems);
		std::cout << '\r' << "Test output data read.\n";
	}

	NeuralNetwork nn({28*28, 28*28, 28*28, 28*28, 10});
	NeuralNetworkTrainer nnt(10, 1e-6, 1e-3, 10);

	std::cout << "Training neural network...";
	const auto p = nnt.trainNeuralNetwork(nn, trainInput, trainOutput);
	std::cout << "\nTraining complete.\n";
	std::cout << "Steps taken: " << p.first << '\n';
	std::cout << "Final cost function difference: " << p.second << '\n';

	std::cout << "Calculating accurary...";
	Eigen::MatrixXd normTestInput = testInput;
	nnt.normalizeFeatures(normTestInput);
	double accurary = calculateAccuracy(nn, normTestInput, testOutput);

	std::cout << "\nThe neural network correctly identified " << accurary << "% of the hand-written digits.\n";

	return 0;
}

double calculateAccuracy(const NeuralNetwork& nn, const Eigen::MatrixXd& input,
	const Eigen::MatrixXd& correctOutput)
{
	const Eigen::MatrixXd output = nn.forwardPropagate(input);
	std::vector<int> maxByIndex;
	maxByIndex.reserve(output.cols());
	for (int j = 0; j < output.cols(); ++j) {
		int maxIndex = 0;
		double maxCoeff = output(0, j);
		for (int i = 1; i < output.rows(); ++i) {
			if (output(i, j) > maxCoeff) {
				maxIndex = i;
				maxCoeff = output(i, j);
			}
		}
		maxByIndex.push_back(maxIndex);
	}

	double accuracy = 0;
	for (int j = 0; j < correctOutput.cols(); ++j) {
		if (correctOutput(maxByIndex[j], j) == 1) {
			++accuracy;
		}
	}
	
	return accuracy / correctOutput.cols();
}

Eigen::MatrixXd readData(std::ifstream& file, const DataType dt, const std::int32_t maxItems)
{
	Eigen::MatrixXd data;

	try {
		file.exceptions(std::ifstream::badbit | std::ifstream::failbit);

		std::int32_t magicNum = 0;
		file.read((char*)&magicNum, sizeof(magicNum));
		// Read as big-endian. Intel processors use little-endian
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

