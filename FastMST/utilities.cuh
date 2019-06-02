#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Graph.h"
#include "CompactGraph.h"
#include "DataStructureOnGpu.cuh"

int createMask(int start, int width);

void minOutgoingEdge(DatastructuresOnGpu onGPU);


void* moveToGpu(void* src, int sizeinBytes);

void* moveToGpu(int* src, int size);

void* moveToGpu(std::vector<int> src);


int* MarkEdgeSegments(DatastructuresOnGpu onGPU);
void MarkEdgeSegmentsOnGpu(DatastructuresOnGpu onGPU, int* flags);

void segmentedScan(int* out, int* in, int * flags, int width);
void segmentedScanInCuda(int* out, int* in, int * flags, int width);

void segmentedMinScan(int* out, int* in, int* flags, int width);
void segmentedMinScanInCuda(int* out, int* in, int* flags, int width);

void split(int* data, int* flags, int width);
void splitInCuda(int* data, int* flags, int width);

__global__ void fill(int* out, int immediate, int width);

__global__ void fill(int* out, int* src, int width, int mask);

__global__ void fill(int* out, int* src, int width, int mask, int from);

__global__ void mark_edge_ptr(int* out, int* ptr, int width);
__global__ void getMinNodes();