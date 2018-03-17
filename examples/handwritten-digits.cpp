#pragma once

#include "NeuralNetwork.h"
#include "NeuralNetworkTrainer.h"

#include "Eigen\Dense"

#include <iostream>
#include <fstream>

enum class DataType {Images, Labels};

Eigen::MatrixXd readData(std::ifstream& file, DataType dt);

int main()
{
	std::ifstream file("../testdata/train-images.idx3-ubyte", std::ios::binary);
	if (!file) {
		std::cout << "Couldnt open data file";
	}

	Eigen::MatrixXd trainImages = readData(file, DataType::Images);
	

	return 0;
}

Eigen::MatrixXd readData(std::ifstream& file, DataType dt)
{
	Eigen::MatrixXd data;

	try {
		file.exceptions(std::ifstream::badbit | std::ifstream::failbit);

		std::int32_t magicNum = 0;
		file.read((char*)&magicNum, 4);
		std::cout << std::hex << magicNum << std::endl;

		std::int32_t nitems = 0;
		file.read((char*)&nitems, 4);
		std::cout << nitems << std::endl;

		std::int32_t nrows = 0, ncols = 0;
		if (dt == DataType::Images) {
			file.read((char*)&nrows, 4);
			file.read((char*)&ncols, 4);
			std::cout << nitems << std::endl;
			std::cout << nitems << std::endl;
		}
		else /*if (dt == DataType::Labels) */ {
			nrows = 10;
			ncols = 1;
		}
		data.resize(nrows, ncols*nitems);
		data.setZero();
		
		unsigned char element;
		for (int k = 0; k < nitems; ++k) {
			for (int i = 0; i < nrows; ++i) {
				for (int j = 0; j < ncols; ++j) {
					file.read((char*)&element, sizeof(element));
					data(i, j + k * ncols) = element;
				}
			}
			if (k % 1000 == 0) {
				std::cout << k << " items read.\n";
			}
		}
	}
	catch (std::ifstream::failure&) {
		std::cout << "Error reading file\n";
	}

	return data;
}

