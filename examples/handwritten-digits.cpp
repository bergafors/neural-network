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
	const std::int32_t maxTrainItems = 10;
	const std::int32_t maxTestItems = 2;

	std::ifstream trainInputFile("../testdata/train-images.idx3-ubyte", std::ios::binary);
	if (!trainInputFile) {
		std::cout << "Couldnt open data file";
	}
	Eigen::MatrixXd trainInput = readData(trainInputFile, DataType::Images, maxTrainItems);

	std::ifstream trainOutputFile("../testdata/train-labels.idx1-ubyte", std::ios::binary);
	if (!trainOutputFile) {
		std::cout << "Couldnt open data file";
	}
	Eigen::MatrixXd trainOutput = readData(trainOutputFile, DataType::Labels, maxTrainItems);

	std::ifstream testInputFile("../testdata/t10k-images.idx3-ubyte", std::ios::binary);
	if (!testInputFile) {
		std::cout << "Couldnt open data file";
	}
	Eigen::MatrixXd testInput = readData(testInputFile, DataType::Images, maxTestItems);

	std::ifstream testOutputFile("../testdata/t10k-labels.idx1-ubyte", std::ios::binary);
	if (!testOutputFile) {
		std::cout << "Couldnt open data file";
	}
	Eigen::MatrixXd testOutput = readData(testOutputFile, DataType::Labels, maxTestItems);

	NeuralNetwork nn({28*28, 28*28, 10});
	NeuralNetworkTrainer nnt(0, 1e-2, 1e-5, 50);
	nnt.trainNeuralNetwork(nn, trainInput, trainOutput);

	double accurary = calculateAccuracy(nn, testInput, testOutput);

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

		std::cout << std::hex << magicNum << std::dec << std::endl;

		std::int32_t nitems = 0;
		file.read((char*)&nitems, sizeof(nitems));
		nitems = _byteswap_ulong(nitems);

		std::cout << nitems << std::endl;


		std::int32_t nrows = 0, ncols = 0;
		if (dt == DataType::Images) {
			file.read((char*)&nrows, sizeof(nrows));
			file.read((char*)&ncols, sizeof(ncols));
			nrows = _byteswap_ulong(nrows);
			ncols = _byteswap_ulong(ncols);
			std::cout << nrows << std::endl;
			std::cout << ncols << std::endl;

		}
		else /*if (dt == DataType::Labels) */ {
			nrows = 10;
			ncols = 1;
		}
		data.resize(nrows*ncols, nitems);
		data.setZero();

		if (maxItems > 0 && nitems > maxItems) {
			nitems = maxItems;
		}
		
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
			int divisor = nitems / 100;
			if (divisor > 0 && (k + 1) % (nitems / 100) == 0) {
				std::cout << '\r' << ((k + 1)*100) / nitems << "% of items read." << std::flush;
			}
		}
		std::cout << std::endl;
	}
	catch (std::ifstream::failure&) {
		std::cout << "Error reading file\n";
	}

	return data;
}

