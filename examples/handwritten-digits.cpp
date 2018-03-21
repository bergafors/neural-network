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
	const std::int32_t maxTrainItems = 100;
	const std::int32_t maxTestItems = 20;

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

	/*std::cout << "Training output:\n";
	for (int i = 0; i < trainOutput.rows(); ++i) {
		std::cout << "Num " << i << ": " << trainOutput.row(i).mean() << std::endl;;
	}

	std::cout << "Test output:\n";
	for (int i = 0; i < testOutput.rows(); ++i) {
		std::cout << "Num " << i << ": " << testOutput.row(i).mean() << std::endl;
	}*/

	/*for (int k = 0; k < 5; ++k) {
		Eigen::MatrixXd::Index ind = 0;
		trainOutput.col(k).maxCoeff(&ind);
		std::cout << "Label: " << ind << std::endl;

		for (int i = 0; i < 28; ++i) {
			for (int j = 0; j < 28; ++j) {
				int val = trainInput(i * 28 + j, k);
				std::cout << (val > 0 ? 1 : 0) << ' ';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}*/
	

	NeuralNetwork nn({28*28, 100, 10});
	for (auto& w : nn.getWeights()) {
		1e-2*w.setRandom();
	}

	NeuralNetworkTrainer nnt(0, 1e-2, 1e-5, 20);

	std::cout << "Normalizing training and test input features...\n";
	const auto normMat = nnt.normalizeFeatures(trainInput);
	std::cout << "Normalization complete.\n";
	if (trainInput.hasNaN()) {
		std::cout << "Error: normalized training input contains NaN values\n";
	}
	if (!trainInput.allFinite()) {
		std::cout << "Error: normalized training input contains Inf values\n";
	}

	const auto& meanMat = normMat.first;
	const auto& invStdDevMat = normMat.second;
	const auto nr = testInput.rows();
	const auto nc = testInput.cols();
	auto normTestInput = (testInput - meanMat.block(0, 0, nr, nc)).cwiseProduct(invStdDevMat.block(0, 0, nr, nc));
	if (normTestInput.hasNaN()) {
		std::cout << "Error: normalized test input contains NaN values\n";
	}
	if (!normTestInput.allFinite()) {
		std::cout << "Error: normalized test input contains Inf values\n";
	}

	std::cout << "Training neural network...\n";
	const auto p = nnt.trainNeuralNetwork(nn, trainInput, trainOutput);
	std::cout << "Training complete.\n";
	std::cout << "Steps taken: " << p.first << '\n';
	std::cout << "Final cost function difference: " << p.second << '\n';

	std::cout << "Calculating accuracy...\n";
	double accuracy = calculateAccuracy(nn, trainInput, trainOutput);

	std::cout << "The neural network correctly identified " << 100*accuracy << "% of the hand-written digits.\n";

	return 0;
}

double calculateAccuracy(const NeuralNetwork& nn, const Eigen::MatrixXd& input,
	const Eigen::MatrixXd& correctOutput)
{
	const Eigen::MatrixXd output = nn.forwardPropagate(input);

	double accuracy = 0;
	Eigen::MatrixXd::Index indexOfMax = 0;
	Eigen::MatrixXd::Index correctIndexOfMax = 0;
	for (int i = 0; i < output.cols(); ++i) {
		output.col(i).maxCoeff(&indexOfMax);
		correctOutput.col(i).maxCoeff(&correctIndexOfMax);

		if (indexOfMax == correctIndexOfMax) {
			++accuracy;
		}
	}

	return accuracy / output.cols();
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

