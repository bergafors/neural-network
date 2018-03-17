#pragma once

#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

#include "Eigen\Dense"

#include <iostream>
#include <fstream>

#include <intrin.h>

enum class DataType {Images, Labels};

Eigen::MatrixXd readData(std::ifstream& file, DataType dt, const std::int32_t maxItems = -1);

int main()
{
	const std::int32_t maxTrainItems = 10000;
	const std::int32_t maxTestItems = 2000;

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

	return 0;
}

Eigen::MatrixXd readData(std::ifstream& file, DataType dt, const std::int32_t maxItems)
{
	Eigen::MatrixXd data;

	try {
		file.exceptions(std::ifstream::badbit | std::ifstream::failbit);

		std::int32_t magicNum = 0;
		file.read((char*)&magicNum, 4);
		magicNum = _byteswap_ulong(magicNum);

		std::int32_t nitems = 0;
		file.read((char*)&nitems, 4);
		nitems = _byteswap_ulong(nitems);

		std::int32_t nrows = 0, ncols = 0;
		if (dt == DataType::Images) {
			file.read((char*)&nrows, 4);
			file.read((char*)&ncols, 4);
			nrows = _byteswap_ulong(nrows);
			ncols = _byteswap_ulong(ncols);
		}
		else /*if (dt == DataType::Labels) */ {
			nrows = 10;
			ncols = 1;
		}
		data.resize(nrows, ncols*nitems);
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
						data(i, j + k * ncols) = element;
					}
					else /*if (dt == DataType::Labels) */ {
						data(element, k*ncols) = 1;
						i = nrows;
					}

				}
			}
			if ((k + 1) % 1000 == 0) {
				std::cout << '\r' << ((k + 1)*100) / nitems << "% of items read.";
			}
		}
		std::cout << std::endl;
	}
	catch (std::ifstream::failure&) {
		std::cout << "Error reading file\n";
	}

	return data;
}

