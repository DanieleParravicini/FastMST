
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include "Graph.h"
#include "CompactGraph.h"
#include <thrust\fill.h>
#include <thrust\partition.h>
#include <thrust\scan.h>


int main()
{
	std::vector<std::vector<int>> weights = { {0,2},{2, 0} };
	Graph g(weights, weights.size());
	std::cout << g.to_string();
	std::cout << g.toCompact().to_string();
	try {
		mst(g);
	}
	catch (...) {
		std::cout << "Some error occurred :'(";
	}

    return 0;
}

void mst(Graph g) {
	CompactGraph c = g.toCompact();

	mst(c);
}

struct compactGraphOnGpu {
	int* vertices = 0;
	int* edgePtr = 0;
	int* weights = 0;
	int* edges = 0;
	int numEdges;
	int numVertices;
	int* X;
};
void mst(CompactGraph g) {
	const int blockFactor = 32;

	compactGraphOnGpu onGPU;
	onGPU.numEdges = g.edges.size();
	onGPU.numVertices = g.vertices.size();

	cudaError_t status;
	try {
		//1. obtain space where to put the data structure that represent the graph.
		status = cudaMalloc(&onGPU.vertices, sizeof(int)*g.vertices.size());
		if (status != cudaError::cudaSuccess)
			throw status;
		status = cudaMalloc(&onGPU.edgePtr, sizeof(int)*g.edgePtr.size());
		if (status != cudaError::cudaSuccess)
			throw status;
		status = cudaMalloc(&onGPU.weights, sizeof(int)*g.weights.size());
		if (status != cudaError::cudaSuccess)
			throw status;
		status = cudaMalloc(&onGPU.edges, sizeof(int)*g.edges.size());
		if (status != cudaError::cudaSuccess)
			throw status;
		//2. transfer data to GPU memory
		status = cudaMemcpy(onGPU.vertices, &g.vertices[0], sizeof(int)*g.vertices.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
		if (status != cudaError::cudaSuccess)
			throw status;
		status = cudaMemcpy(onGPU.edgePtr, &g.edgePtr[0], sizeof(int)*g.edgePtr.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
		if (status != cudaError::cudaSuccess)
			throw status;
		status = cudaMemcpy(onGPU.weights, &g.weights[0], sizeof(int)*g.weights.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
		if (status != cudaError::cudaSuccess)
			throw status;
		status = cudaMemcpy(onGPU.edges, &g.edges[0], sizeof(int)*g.edges.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
		if (status != cudaError::cudaSuccess)
			throw status;
		//3. once memory transfer has been achievede we have to find 
		//   for each vertex the min cost outgoing edge
		minOutgoingEdge(onGPU);
		

		//4. rebuild compact graph representation for next algorithm iteration.


	}
	catch (cudaError_t err) {
		std::cout << err;
		cudaFree(onGPU.vertices);
		cudaFree(onGPU.edgePtr);
		cudaFree(onGPU.weights);
		cudaFree(onGPU.edges);

		throw err;
	}


	cudaFree(onGPU.vertices);
	cudaFree(onGPU.edgePtr);
	cudaFree(onGPU.weights);
	cudaFree(onGPU.edges);
	
}

int* copy() {
	/*
	status = cudaMalloc(&onGPU.vertices, sizeof(int)*g.vertices.size());
	if (status != cudaError::cudaSuccess)
		throw status;
	status = cudaMemcpy(onGPU.vertices, &g.vertices[0], sizeof(int)*g.vertices.size(), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if (status != cudaError::cudaSuccess)
		throw status;
		*/
}
int* MarkEdgeSegments(compactGraphOnGpu onGPU) {
	cudaError_t status;
	int * flags;
	try {
		status = cudaMalloc(&flags, sizeof(int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;
		fill << <onGPU.numEdges / 32, 32 >> >(flags, 0, onGPU.numEdges);
		mark_edge_ptr << <onGPU.numVertices / 32, 32 >> >(flags, onGPU.edgePtr, onGPU.numVertices);
		
	}
	catch (...) {
		cudaFree(flags);
		throw status;
	}
	return flags;
}

void segmentedScan(int* out, int* in, int * flags, int width) {
	int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	thrust::inclusive_scan(tmpKeys, tmpKeys + width, tmpKeys);
	thrust::inclusive_scan_by_key(tmpKeys, tmpKeys + width, in, out);
}

struct min {
	__host__ __device__
	int  operator()(const int a, const int b) const 
	{ 
		return a < b ? a: b; 
	}
};

void segmentedMinScan(int* out, int* in, int* flags, int width) {
	int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	thrust::inclusive_scan(tmpKeys, tmpKeys + width, tmpKeys);

	thrust::equal_to<int> binary_pred;
	min binary_op;

	thrust::inclusive_scan_by_key(tmpKeys, tmpKeys + width, in, out, binary_pred, binary_op);
}

void split(int* data, int* flags, int width) {
	int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	thrust::inclusive_scan(tmpKeys, tmpKeys + width, tmpKeys);
	thrust::sort_by_key(tmpKeys, tmpKeys + width, data);
}

__global__ void fill(int* out, int immediate, int width) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = immediate;
	}
}
__global__ void mark_edge_ptr(int* out, int* ptr, int width) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[ptr[idx]] = 1;
	}
}
__global__ void getMinNodes() {

}