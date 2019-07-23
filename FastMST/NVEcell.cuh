#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utils.h"

struct NVEcell {
	long long unsigned int cell;

	__host__ __device__ void setSource(unsigned int s);
	__host__ __device__ void setDestination(unsigned int d);
	__host__ __device__ void setWeight(unsigned int w);

	__host__ __device__ unsigned int getSource();
	__host__ __device__ unsigned int getDestination();
	__host__ __device__ unsigned int getWeight();

};


__host__ __device__ bool operator < (const NVEcell &lhs, const NVEcell &rhs);