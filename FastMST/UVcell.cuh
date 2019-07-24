#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.h"
#include <thrust\functional.h>

struct UVcell {
	unsigned int UW;
	unsigned int ID;

	__host__ __device__ void setDestination(unsigned int d);
	__host__ __device__ void setWeight(unsigned int w);
	__host__ __device__ void setID(unsigned int ID);

	__host__ __device__ unsigned int getDestination();
	__host__ __device__ unsigned int getWeight();
	__host__ __device__ unsigned int getID();
};


