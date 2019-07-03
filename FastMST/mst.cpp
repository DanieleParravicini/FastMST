#include "mst.h"

int mst(Graph &g) {

	CompactGraph c(g);
	return mst(c);

}

int mst(CompactGraph g) {

	DatastructuresOnGpu onGPU;
	onGPU.cost = 0;
	onGPU.numEdges = g.edges.size();
	onGPU.numVertices = g.vertices.size();
	int res = -1;

	try {
		//1. move data structures to GPU memory
		onGPU.vertices = (unsigned int*)moveToGpu(g.vertices);
		onGPU.edgePtr = (unsigned int*)moveToGpu(g.edgePtr);
		onGPU.edges = (unsigned int*)moveToGpu(g.edges);
		onGPU.weights = (unsigned int*)moveToGpu(g.weights);

		cudaError_t status;
		status = cudaMalloc(&onGPU.X, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.F, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.S, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.C, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		res = mst(&onGPU);

		cudaFree(onGPU.vertices);
		cudaFree(onGPU.edgePtr);
		cudaFree(onGPU.weights);
		cudaFree(onGPU.edges);
		cudaFree(onGPU.X);
		cudaFree(onGPU.F);
		cudaFree(onGPU.S);
		cudaFree(onGPU.C);
	}
	catch (cudaError_t err) {
		std::cout << err;
		cudaFree(onGPU.vertices);
		cudaFree(onGPU.edgePtr);
		cudaFree(onGPU.weights);
		cudaFree(onGPU.edges);
		cudaFree(onGPU.X);
		cudaFree(onGPU.F);
		cudaFree(onGPU.S);
		cudaFree(onGPU.C);
		throw err;
	}
	catch (thrust::system_error &e)
	{
		std::cerr << "CUDA error:" << e.what() << std::endl;
		//got some unspecified launch failure?
		//cause could be a watchdog inserted by windows OS
		// for additional details and workaround follow the next link:
		//https://docs.nvidia.com/gameworks/content/developertools/desktop/timeout_detection_recovery.htm
		cudaFree(onGPU.vertices);
		cudaFree(onGPU.edgePtr);
		cudaFree(onGPU.weights);
		cudaFree(onGPU.edges);
		cudaFree(onGPU.X);
		cudaFree(onGPU.F);
		cudaFree(onGPU.S);
		cudaFree(onGPU.C);

	}

	return res;
}

int mst(DatastructuresOnGpu *onGPU) {
	int iter = 0;
	while (onGPU->numVertices > 1) {

		//print results

#ifdef DEBUG
		std::cout << std::endl << "[iter " << iter << "]" << std::endl;
		std::cout << "Vertices" << std::endl;
		debug_device_ptr(onGPU->vertices, onGPU->numVertices);
		std::cout << "Edge ptr" << std::endl;
		debug_device_ptr(onGPU->edgePtr, onGPU->numVertices);
		std::cout << "Edges" << std::endl;
		debug_device_ptr(onGPU->edges, onGPU->numEdges);
		std::cout << "weights" << std::endl;
		debug_device_ptr(onGPU->weights, onGPU->numEdges);

		onGPU->printForWebgraphvizrint();
#endif

		//1. for each vertex we have to find 
		//   the min cost outgoing edge
		minOutgoingEdge(onGPU);
		//2. Build initial version of successor vector
		//	 and update the cost of MST.
		buildSuccessorAndUpdateCosts(onGPU);
		//3. rebuild compact graph representation for next algorithm iteration.
		rebuildConsedGraphrepresentation(onGPU);
		//4. update num of vertices, num of edges counter using scan result.
		iter++;

	}
	return onGPU->cost;
}

