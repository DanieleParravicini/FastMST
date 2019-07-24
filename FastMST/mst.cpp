#include "mst.h"
#include "cuda_profiler_api.h"


long long int mst(Graph &g, std::vector<Edge> &mst_result) {
	ExpandedGraph c(g );
	
	unsigned int* array_mst_result = (unsigned int *) malloc(sizeof(unsigned int) * c.vertices.size() * 2);
	unsigned int length = 0;

	long long int ret = mst(c, array_mst_result, &length);

	for (int i = 0; i < length; i++) {

		std::vector<unsigned int>::iterator pos = std::lower_bound(c.edgePtr.begin(), c.edgePtr.end(), array_mst_result[i]);

		int s = pos - c.edgePtr.begin();
		if (pos == c.edgePtr.end() || *pos != array_mst_result[i]){
			s -= 1;
		}
		int d = c.edges[array_mst_result[i]];

		std::pair<Edge, bool> res = boost::edge(s, d, g);
		assert(res.second);
		mst_result.push_back(res.first);
	}
	free(array_mst_result);
	return ret;
}

long long int mst(ExpandedGraph &g, unsigned int * array_mst_result, unsigned int * lenght_array_mst_result) {
	cudaError_t status;
	DatastructuresOnGpu onGPU;
	onGPU.cost = 0;
	onGPU.savedEdges = 0;
	onGPU.numEdges = g.edges.size();
	onGPU.numVertices = g.vertices.at(g.vertices.size() - 1)+1;
	long long int res = -1;
	
	try {
		//0. obtain max shared memory exploitable per block
		cudaDeviceGetAttribute(&onGPU.maxSharedBytes, cudaDevAttrMaxSharedMemoryPerBlock, 0);
		//1. move data structures to GPU memory
		//create vertices as large as the number of edges since it will be reused in the reconstruction of the graph.
		status = cudaMalloc(&onGPU.vertices, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMemcpy(onGPU.vertices, &g.vertices[0], sizeof(unsigned int)*onGPU.numEdges, cudaMemcpyHostToDevice);
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

		status = cudaMalloc(&onGPU.NVE, sizeof(NVEcell)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.edgeID, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.edgeIDresult, sizeof(unsigned int)*onGPU.numVertices*2);
		if (status != cudaError::cudaSuccess)
			throw status;
		
		cudaMemcpy(onGPU.edgeID, &g.edgesIds[0], sizeof(unsigned int)*onGPU.numEdges, cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaProfilerStart();
		cudaEventRecord(start);

		res = mst(&onGPU);

		cudaEventRecord(stop);
		cudaProfilerStop();
		
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		std::cout << "Time gpu occupied: "<< milliseconds << " [ms]"<< std::endl;

		cudaMemcpy(array_mst_result, onGPU.edgeIDresult, sizeof(unsigned int)* onGPU.savedEdges, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		*lenght_array_mst_result = 0;
		for (int i = 0; i < onGPU.savedEdges; i++) {
			if (array_mst_result[i] < g.edges.size()) {
				array_mst_result[*lenght_array_mst_result] = array_mst_result[i];
				(*lenght_array_mst_result)++;
			}
		}


		cudaFree(onGPU.vertices);
		cudaFree(onGPU.edgePtr);
		cudaFree(onGPU.weights);
		cudaFree(onGPU.edges);
		cudaFree(onGPU.X);
		cudaFree(onGPU.F);
		cudaFree(onGPU.S);
		cudaFree(onGPU.NVE);
		cudaFree(onGPU.edgeID);
		cudaFree(onGPU.edgeIDresult);
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
		cudaFree(onGPU.NVE);
		cudaFree(onGPU.edgeID);
		cudaFree(onGPU.edgeIDresult);

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
		cudaFree(onGPU.NVE);
		cudaFree(onGPU.edgeID);
		cudaFree(onGPU.edgeIDresult);

		throw e;
	}

	return res;
}

long long int mst(DatastructuresOnGpu *onGPU) {
	int iter = 0;
	while (onGPU->numVertices > 1) {
#if defined INFORMATIVE
		std::cout << "[ iter " << iter << " ]" << std::endl;
		//print results
#endif
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

		onGPU->printForWebgraphviz();

#endif

		//1. for each vertex we have to find 
		//   the min cost outgoing edge
		minOutgoingEdge(onGPU);
		//2. Build initial version of successor vector
		//	 and update the cost of MST.
		moveMinWeightsAndSuccessor(onGPU);
		computeCosts(onGPU);
		buildSuccessor(onGPU);
		saveMinOutgoingEdges(onGPU);
		
		//3. rebuild compact graph representation for next algorithm iteration.
		buildSupervertexId(onGPU);
		orderUVW(onGPU);
		rebuildEdgeWeights(onGPU);
		rebuildEdgePtr( onGPU);

		//4. update num of vertices, num of edges counter using scan result.
		iter++;
		onGPU->numVertices = onGPU->newNumVertices;
		onGPU->numEdges = onGPU->newNumEdges;

	}
	return onGPU->cost;
}

long long int verifyMst(Graph &g, std::vector<Edge> &mst_result) {

	ExpandedGraph c(g);

	unsigned int* array_mst_result = (unsigned int *)malloc(sizeof(unsigned int) * c.vertices.size() * 2);
	unsigned int length = 0;
	
	long long int ret = verifyMst(c, array_mst_result, &length);
	
	for (int i = 0; i < length; i++) {

		std::vector<unsigned int>::iterator pos = std::lower_bound(c.edgePtr.begin(), c.edgePtr.end(), array_mst_result[i]);

		int s = pos - c.edgePtr.begin();
		if (pos == c.edgePtr.end() || *pos != array_mst_result[i]) {
			s -= 1;
		}
		int d = c.edges[array_mst_result[i]];

		std::pair<Edge, bool> res = boost::edge(s, d, g);
		assert(res.second);
		mst_result.push_back(res.first);
	}

	free(array_mst_result);
	return ret;
}

long long int verifyMst(ExpandedGraph &g, unsigned int * array_mst_result, unsigned int * lenght_array_mst_result) {
	cudaError_t status;
	DatastructuresOnGpu onGPU;
	onGPU.cost = 0;
	onGPU.savedEdges = 0;
	onGPU.numEdges = g.edges.size();
	onGPU.numVertices = g.vertices.at( g.vertices.size()-1) + 1;
	long long int res = -1;
	
	try {
		//0. obtain max shared memory exploitable per block
		cudaDeviceGetAttribute(&onGPU.maxSharedBytes, cudaDevAttrMaxSharedMemoryPerBlock, 0);
		//1. move data structures to GPU memory
		//create vertices as large as the number of edges since it will be reused in the reconstruction of the graph.
		status = cudaMalloc(&onGPU.vertices, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMemcpy(onGPU.vertices, &g.vertices[0], sizeof(unsigned int)*onGPU.numEdges, cudaMemcpyHostToDevice);
		if (status != cudaError::cudaSuccess)
			throw status;

		onGPU.edgePtr = (unsigned int*)moveToGpu(g.edgePtr);
		onGPU.edges = (unsigned int*)moveToGpu(g.edges);
		onGPU.weights = (unsigned int*)moveToGpu(g.weights);

		status = cudaMalloc(&onGPU.X, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.F, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.S, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.NVE, sizeof(NVEcell)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.edgeID, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;

		status = cudaMalloc(&onGPU.edgeIDresult, sizeof(unsigned int)*onGPU.numVertices * 2);
		if (status != cudaError::cudaSuccess)
			throw status;

		cudaMemcpy(onGPU.edgeID, &g.edgesIds[0] , sizeof(unsigned int)* onGPU.numEdges, cudaMemcpyHostToDevice);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);


		cudaProfilerStart();
		cudaEventRecord(start);

		res = verifyMst(&onGPU);

		cudaEventRecord(stop);
		cudaProfilerStop();

		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		std::cout << "Time gpu occupied: " << milliseconds << " [ms]" << std::endl;

		cudaMemcpy(array_mst_result, onGPU.edgeIDresult, sizeof(unsigned int)* onGPU.savedEdges, cudaMemcpyDeviceToHost);

		long long unsigned cost;
		*lenght_array_mst_result = 0;
		for (int i = 0 ; i < onGPU.savedEdges; i++) {
			if (array_mst_result[i] < g.edges.size()) {
				array_mst_result[*lenght_array_mst_result] = array_mst_result[i];
				*lenght_array_mst_result += 1;
				cost += g.weights[array_mst_result[i]];
			}
		}
		
		assert(cost == res);

		cudaFree(onGPU.vertices);
		cudaFree(onGPU.edgePtr);
		cudaFree(onGPU.weights);
		cudaFree(onGPU.edges);
		cudaFree(onGPU.X);
		cudaFree(onGPU.F);
		cudaFree(onGPU.S);
		cudaFree(onGPU.NVE);
		cudaFree(onGPU.edgeID);
		cudaFree(onGPU.edgeIDresult);

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
		cudaFree(onGPU.NVE);
		cudaFree(onGPU.edgeID);
		cudaFree(onGPU.edgeIDresult);

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
		cudaFree(onGPU.NVE);
		cudaFree(onGPU.edgeID);
		cudaFree(onGPU.edgeIDresult);

		throw e;
	}

	return res;
}

long long int verifyMst(DatastructuresOnGpu* onGPU) {
	//onGPU->printForWebgraphvizrint();

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

		onGPU->printForWebgraphviz();
#endif

		unsigned long long int curr_cost = onGPU->cost;
		unsigned int currnumVertices = onGPU->numVertices;

		Graph g;
		toGraph(g, onGPU);

		
		std::set<Edge> edges;
	
		boost::kruskal_minimum_spanning_tree(g, std::inserter(edges, edges.end()));

		unsigned long long int costSubProblem = 0;
		for (Edge e : edges) {
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
		saveMinOutgoingEdges(onGPU);

		//3. rebuild compact graph representation for next algorithm iteration.
		buildSupervertexId(onGPU);
		orderUVW(onGPU);
		rebuildEdgeWeights(onGPU);
		rebuildEdgePtr(onGPU);
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
