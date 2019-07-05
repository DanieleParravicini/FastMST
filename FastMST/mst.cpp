#include "mst.h"
#include "cuda_profiler_api.h"

int mst(Graph &g) {

	CompactGraph c(g);
	return mst(c);

}

int mst(CompactGraph &g) {
	cudaError_t status;
	DatastructuresOnGpu onGPU;
	onGPU.cost = 0;
	onGPU.numEdges = g.edges.size();
	onGPU.numVertices = g.vertices.size();
	int res = -1;

	try {
		//1. move data structures to GPU memory
		//create vertices as large as the number of edges since it will be reused in the reconstruction of the graph.
		status = cudaMalloc(&onGPU.vertices, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMemcpy(onGPU.vertices, &g.vertices[0], sizeof(unsigned int)*onGPU.numVertices, cudaMemcpyHostToDevice);
		if (status != cudaError::cudaSuccess)
			throw status;

				
		onGPU.edgePtr	= (unsigned int*)moveToGpu(g.edgePtr);
		onGPU.edges		= (unsigned int*)moveToGpu(g.edges);
		onGPU.weights	= (unsigned int*)moveToGpu(g.weights);

		
		status = cudaMalloc(&onGPU.X, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.F, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.S, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		
		cudaProfilerStart();
		res = verifyMst(&onGPU);
		cudaProfilerStop();

		cudaFree(onGPU.vertices);
		cudaFree(onGPU.edgePtr);
		cudaFree(onGPU.weights);
		cudaFree(onGPU.edges);
		cudaFree(onGPU.X);
		cudaFree(onGPU.F);
		cudaFree(onGPU.S);
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
		throw e;
	}

	return res;
}

int mst(DatastructuresOnGpu *onGPU) {
	int iter = 0;
	while (onGPU->numVertices > 1) {
		std::cout <<"[ iter " << iter << " ]"<< std::endl;
		//print results

#if defined(PEDANTIC) || defined(DEBUG)
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
		moveMinWeightsAndSuccessor(onGPU);
		computeCosts(onGPU);
		buildSuccessor(onGPU);
		
		//3. rebuild compact graph representation for next algorithm iteration.
		buildSupervertexId(onGPU);
		orderUVW(onGPU);
		rebuildEdgeWeights(onGPU);
		rebuildEdgePtr( onGPU);
		rebuildVertices( onGPU);
		//4. update num of vertices, num of edges counter using scan result.
		iter++;
		onGPU->numVertices = onGPU->newNumVertices;
		onGPU->numEdges = onGPU->newNumEdges;

	}
	return onGPU->cost;
}

int verifyMst(DatastructuresOnGpu* onGPU) {

	if (onGPU->numVertices <= 1)
		return onGPU->cost;

#if defined(PEDANTIC) || defined(DEBUG)
		//print results
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

		int curr_cost = onGPU->cost;
		int currnumVertices = onGPU->numVertices;

		Graph g;
		toGraph(g, onGPU);
		struct InSpanning {
			std::set<Edge> edges;
			bool operator()(Edge e) const { return edges.count(e) > 0; }
		} spanning;

		boost::kruskal_minimum_spanning_tree(g, std::inserter(spanning.edges, spanning.edges.end()));

		int costSubProblem = 0;
		for (Edge e : spanning.edges) {
			costSubProblem += boost::get(boost::edge_weight, g, e);
		}

		//1. for each vertex we have to find 
		//   the min cost outgoing edge
		minOutgoingEdge(onGPU);
		//2. Build initial version of successor vector
		//	 and update the cost of MST.
		moveMinWeightsAndSuccessor(onGPU);
		computeCosts(onGPU);
		buildSuccessor(onGPU);

		//3. rebuild compact graph representation for next algorithm iteration.
		buildSupervertexId(onGPU);
		orderUVW(onGPU);
		rebuildEdgeWeights(onGPU);
		rebuildEdgePtr(onGPU);
		rebuildVertices(onGPU);
		//4. update num of vertices, num of edges counter using scan result.
		
		//5. solve sub problem mst
		onGPU->numVertices = onGPU->newNumVertices;
		onGPU->numEdges = onGPU->newNumEdges;
		

		verifyMst(onGPU);

		if (onGPU->cost - curr_cost == costSubProblem) {
			std::cout << "[ #vertices " << currnumVertices << " ] OK cost:" << costSubProblem << std::endl;
		}
		else {
			std::cout << "[ #vertices " << currnumVertices << " ] KO" << "[ boruvska "<< onGPU->cost - curr_cost <<"] [ golden model "<< costSubProblem <<"]" << std::endl;
		}


	return onGPU->cost;
}