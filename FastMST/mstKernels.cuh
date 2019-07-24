#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DataStructureOnGpu.h"
#include "NVEcell.cuh"
#include "UVcell.cuh"
#include "Utils.h"

#include <algorithm>

#include <thrust\device_vector.h>
#include <thrust\scan.h>
#include <thrust\sort.h>

__global__ void moveWeightsAndSuccessors(unsigned int* src, unsigned int* dstW, unsigned int * dstS, unsigned int n);


unsigned int* MarkEdgeSegments(DatastructuresOnGpu onGPU);

void MarkEdgeSegmentsOnGpu(DatastructuresOnGpu onGPU, unsigned int* flags);
__global__ void mark_edge_ptr(unsigned int* out, unsigned int* ptr, unsigned int width);


void minOutgoingEdge(DatastructuresOnGpu* onGPU);

void moveMinWeightsAndSuccessor(DatastructuresOnGpu* onGPU);
void computeCosts(DatastructuresOnGpu* onGPU);
void buildSuccessor(DatastructuresOnGpu* onGPU);
void saveMinOutgoingEdges(DatastructuresOnGpu* onGPU);

void orderUVW(DatastructuresOnGpu*onGPU);
void rebuildEdgePtr(DatastructuresOnGpu* onGPU);
void rebuildEdgeWeights(DatastructuresOnGpu* onGPU);

void buildSupervertexId(DatastructuresOnGpu* onGPU);
void rebuildCompressedGraphRepresentation(DatastructuresOnGpu* onGPU);