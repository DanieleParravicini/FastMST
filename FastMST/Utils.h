#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DataStructureOnGpu.h"


#include <thrust\device_vector.h>
#include <thrust\scan.h>
#include <thrust\sort.h>

#define BLOCK_SIZE 16
#define WEIGHT_SIZE 10
#define VERTEX_SIZE 22
//#define DEBUG

__host__ __device__ unsigned int createMask(int start, int width);

void* moveToGpu(void* src, int sizeinBytes);
void* moveToGpu(int* src, int size);
void* moveToGpu(std::vector<int> src);

void debug_device_ptr(unsigned int* ptr, unsigned int items);
void debug_device_ptr(unsigned int* ptr, unsigned int items, unsigned int nrbit);

unsigned int* MarkEdgeSegments(DatastructuresOnGpu onGPU);

void MarkEdgeSegmentsOnGpu(DatastructuresOnGpu onGPU, unsigned int* flags);

void segmentedScan(int* out, int* in, int * flags, int width);
void segmentedScanInCuda(unsigned int* out, unsigned int* in, unsigned int * flags, unsigned int width);

void segmentedMinScan(int* out, int* in, int* flags, int width);
void segmentedMinScanInCuda(unsigned int* out, unsigned int* in, unsigned int* flags, unsigned int width);

void split(int* data, int* flags, int width);
void splitInCuda(int* data, int* flags, int width);

__global__ void fill(unsigned int* out, unsigned int immediate, unsigned int width);

__global__ void fill(unsigned int* out, unsigned int* src, unsigned int width, unsigned int mask);

__global__ void fill(unsigned int* out, unsigned int* src, unsigned int width, unsigned int mask, unsigned int from);

__global__ void mark_edge_ptr(unsigned int* out, unsigned int* ptr, unsigned int width);
__global__ void getMinNodes();


__global__ void copyIndirected(int* dst, int* src, int * ptr, int n);

template<int mask, int shift>
int unMaskAndShift(int a) {
	return (a & mask) >> shift;
}


template< int op(int)>
__global__ void copy(int* src, int* dst, int n);

__global__ void moveWeightsAndSuccessors(unsigned int* src, unsigned int* dstW, unsigned int * dstS, unsigned int n);


void minOutgoingEdge(DatastructuresOnGpu* onGPU);

void buildSuccessorAndUpdateCosts(DatastructuresOnGpu *onGPU);

void rebuildConsedGraphrepresentation(DatastructuresOnGpu* onGPU);