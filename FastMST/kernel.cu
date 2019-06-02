#include "kernel.h"
#include "cudpp.h"

#define BLOCK_SIZE 16


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

void mst(CompactGraph g) {

	DatastructuresOnGpu onGPU;
	onGPU.numEdges = g.edges.size();
	onGPU.numVertices = g.vertices.size();

	try {
		//1. move data structures to GPU memory
		onGPU.vertices = (int*)moveToGpu(g.vertices);
		onGPU.edgePtr = (int*)moveToGpu(g.edgePtr);
		onGPU.edges = (int*)moveToGpu(g.edges);
		onGPU.weights = (int*)moveToGpu(g.weights);

		mst(onGPU);
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

void mst(DatastructuresOnGpu onGPU) {
	if (onGPU.numVertices == 1)
		return;

	//1. for each vertex we have to find 
	//   the min cost outgoing edge
	minOutgoingEdge(onGPU);


	//3. rebuild compact graph representation for next algorithm iteration.

	//4. update vertices count. by using scan result.
	//4. recall mst();
}



void minOutgoingEdge(DatastructuresOnGpu onGPU) {
	cudaError_t status;
	status = cudaMalloc(&onGPU.X, sizeof(int)*onGPU.numEdges);
	if (status != cudaError::cudaSuccess)
		throw status;

	fill << <onGPU.numEdges / BLOCK_SIZE, BLOCK_SIZE >> >(onGPU.X, onGPU.edges, onGPU.numEdges, createMask(0, 22));
	fill << <onGPU.numEdges / BLOCK_SIZE, BLOCK_SIZE >> >(onGPU.X, onGPU.weights, onGPU.numEdges, createMask(22, 10), 22);

	status = cudaMalloc(&onGPU.F, sizeof(int)*onGPU.numEdges);
	if (status != cudaError::cudaSuccess)
		throw status;
	fill << <onGPU.numEdges / BLOCK_SIZE, BLOCK_SIZE >> >(onGPU.F, 0, onGPU.numEdges);
	mark_edge_ptr << <onGPU.numVertices / BLOCK_SIZE, BLOCK_SIZE >> >(onGPU.F, onGPU.edgePtr, onGPU.numVertices);

	segmentedMinScanInCuda(onGPU.X, onGPU.X, onGPU.F, onGPU.numEdges);
}


int* MarkEdgeSegments(compactGraphOnGpu onGPU) {
	cudaError_t status;
	int * flags;
	try {
		status = cudaMalloc(&flags, sizeof(int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;
		MarkEdgeSegmentsOnGpu(onGPU, flags);

	}
	catch (...) {
		cudaFree(flags);
		throw status;
	}
	return flags;
}

void MarkEdgeSegmentsOnGpu(compactGraphOnGpu onGPU, int* flags) {
	fill << <onGPU.numEdges / BLOCK_SIZE, BLOCK_SIZE >> >(flags, 0, onGPU.numEdges);
	mark_edge_ptr << <onGPU.numVertices / BLOCK_SIZE, BLOCK_SIZE >> >(flags, onGPU.edgePtr, onGPU.numVertices);
}