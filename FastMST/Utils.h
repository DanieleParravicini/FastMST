#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DataStructureOnGpu.h"
#include <algorithm>

#include <thrust\device_vector.h>
#include <thrust\scan.h>
#include <thrust\sort.h>

#define BLOCK_SIZE  32
#define WEIGHT_SIZE 13
#define VERTEX_SIZE 19
#define SHARED_MEMORY_LIMIT 48000
//#define DEBUG
//#define PEDANTIC

__host__ __device__ unsigned int createMask(int start, int width);

void* moveToGpu(void* src, size_t sizeinBytes);
void* moveToGpu(unsigned int* src, int size);
void* moveToGpu(std::vector<unsigned int> src);

void debug_device_ptr(unsigned int* ptr, unsigned int items);
void debug_device_ptr(unsigned int* ptr, unsigned int items, unsigned int nrbit);

unsigned int* MarkEdgeSegments(DatastructuresOnGpu onGPU);

void MarkEdgeSegmentsOnGpu(DatastructuresOnGpu onGPU, unsigned int* flags);


void segmentedMinScan(int* out, int* in, int* flags, int width);
void segmentedMinScanInCuda(unsigned int* out, unsigned int* in, unsigned int* flags, unsigned int width);

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

void moveMinWeightsAndSuccessor(DatastructuresOnGpu* onGPU);
void computeCosts(DatastructuresOnGpu* onGPU);
void buildSuccessor(DatastructuresOnGpu* onGPU);

void orderUVW(DatastructuresOnGpu*onGPU);
void rebuildEdgePtr(DatastructuresOnGpu* onGPU);
void rebuildVertices(DatastructuresOnGpu* onGPU);
void rebuildEdgeWeights(DatastructuresOnGpu* onGPU);

void buildSupervertexId(DatastructuresOnGpu* onGPU);
void rebuildCompressedGraphRepresentation(DatastructuresOnGpu* onGPU);