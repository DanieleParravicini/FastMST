#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DataStructureOnGpu.h"
#include "NVEcell.cuh"
#include <algorithm>

#include <thrust\device_vector.h>
#include <thrust\scan.h>
#include <thrust\sort.h>

#define BLOCK_SIZE  32
#define WEIGHT_SIZE 10
#define VERTEX_SIZE 22

//#define DEBUG
//#define PEDANTIC
//#define THRUST_DEBUG

__host__ __device__ unsigned int createMask(int start, int width);

void* moveToGpu(void* src, size_t sizeinBytes);
void* moveToGpu(unsigned int* src, int size);
void* moveToGpu(std::vector<unsigned int> src);

void debug_device_ptr(unsigned int* ptr, unsigned int items);
void debug_device_ptr(unsigned int* ptr, unsigned int items, unsigned int nrbit);

void segmentedMinScan(int* out, int* in, int* flags, int width);
void segmentedMinScanInCuda(unsigned int* out, unsigned int* in, unsigned int* flags, unsigned int width);

int grid(int n, int block);


__global__ void fill(unsigned int* out, unsigned int immediate, unsigned int width);
__global__ void fill(unsigned int* out, unsigned int* src, unsigned int width, unsigned int mask);
__global__ void fill(unsigned int* out, unsigned int* src, unsigned int width, unsigned int mask, unsigned int from);


__global__ void copyIndirected(unsigned int* dst, unsigned int* src, unsigned int * ptr, unsigned int n, unsigned int m);

