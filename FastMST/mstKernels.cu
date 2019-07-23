#include "mstKernels.cuh"

__global__ void moveWeightsAndSuccessors(unsigned int* src, unsigned int* dstW, unsigned int * dstS, unsigned int n) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < n) {
		int successor = src[idx] & createMask(0, VERTEX_SIZE);
		int successor_of_successor = src[successor] & createMask(0, VERTEX_SIZE);

		int weight;
		if (idx < successor && idx == successor_of_successor) {
			weight = 0;
			successor = idx;
		}
		else {
			weight = (src[idx] & createMask(VERTEX_SIZE, WEIGHT_SIZE)) >> VERTEX_SIZE;
		}
		dstW[idx] = weight;
		dstS[idx] = successor;
	}
}


void MarkEdgeSegmentsOnGpu(DatastructuresOnGpu onGPU, unsigned int* flags) {
	cudaMemset(flags, 0, onGPU.numEdges* sizeof(unsigned int));
	mark_edge_ptr <<<  grid(onGPU.numVertices, BLOCK_SIZE), BLOCK_SIZE >> > (flags, onGPU.edgePtr, onGPU.numVertices);
}


__global__ void mark_edge_ptr(unsigned int* out, unsigned int* ptr, unsigned int width) {
	//first idx start at 0, which should not be marked!
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx > 0 && idx < width) {
		out[ptr[idx]] = 1;
	}
}

__global__ void mark_discontinuance(unsigned int* out, unsigned int* ptr, unsigned int width) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx + 1 < width) {
		if (ptr[idx] != ptr[idx + 1]) {
			out[idx + 1] = 1;
		}
	}
}


__global__ void replaceTillFixedPointInShared(unsigned int * S, unsigned int n) {
	//Move S of block into a shared memory.
	extern __shared__ unsigned int A[];
	int pos;
	int items_per_thread = (n + blockDim.x - 1) / blockDim.x;
#ifdef PEDANTIC
	if (threadIdx.x == 0) {
		printf("total : %d elements; %d items for each thread", n, items_per_thread);
	}
#endif
	for (int i = 0; i < items_per_thread; i++) {
		pos = threadIdx.x * items_per_thread + i;
		if (pos < n) {
			A[pos] = S[pos];
		}
	}

	__shared__ bool flag;
	unsigned int new_s;
	unsigned int old_s;
	do {
		__syncthreads();
		if (threadIdx.x == 0) {
			flag = false;
		}

		for (int i = 0; i < items_per_thread; i++) {
			pos = threadIdx.x*items_per_thread + i;
			if (pos < n) {
				old_s = A[pos];
				new_s = A[old_s];


				if (old_s != new_s) {
					flag = true;
					A[pos] = new_s;
				}
			}
			__syncthreads();
		}

	} while (flag);


	for (int i = 0; i < items_per_thread; i++) {
		pos = (threadIdx.x * items_per_thread) + i;
		//printf("[thread : %d of %d ] pos %d \n ", threadIdx.x, blockDim.x, pos);
		if (pos < n) {
			S[pos] = A[pos];
		}
	}


}



__global__ void replaceTillFixedPoint(unsigned int * S, unsigned int n , unsigned int maxShared) {
	//the shared memory is used as a cache
	//first maxShared items are stored in shared memory instead of global memory
	int pos;
	int items_per_thread = (n + blockDim.x - 1) / blockDim.x;
	extern __shared__ unsigned int cache[];

#ifdef PEDANTIC
	if (threadIdx.x == 0) {
		printf("total : %d elements; %d items for each thread", n, items_per_thread);
	}
#endif

	for (int i = 0; i < items_per_thread; i++) {
		pos = threadIdx.x * items_per_thread + i;
		if (pos < maxShared) {
			cache[pos] = S[pos];
		}
	}

	__syncthreads();

	unsigned int new_s;
	unsigned int old_s;
	for (int i = 0; i < items_per_thread; i++) {
		pos = threadIdx.x*items_per_thread + i;
		
		if (pos < n) {

			if (pos < maxShared) {
				new_s = cache[pos];
			}
			else {
				new_s = S[pos];
			}
			
			do {

				old_s = new_s;
				if (old_s < maxShared) {
					new_s = cache[old_s];
				}
				else {
					new_s = S[old_s];
				}

			} while (old_s != new_s);

			
		}
		__syncthreads();

		if (pos < n ) {
			S[pos] = new_s;
		}

		__syncthreads();
	}

}
__global__ void loadNVE(NVEcell * NVE, unsigned int* v, unsigned int * e, unsigned int *w, unsigned int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n) {
		NVE[idx].setSource(v[idx]);
		NVE[idx].setDestination(e[idx]);
		NVE[idx].setWeight(w[idx]);
		//printf("v %d e %d w %d : result %d", v[idx], e[idx], w[idx], NVE[idx].cell);
	}
}

__global__ void unloadNVE(NVEcell * NVE, unsigned int* v, unsigned int * e, unsigned int *w, unsigned int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n) {
		v[idx] = NVE[idx].getSource();
		e[idx] = NVE[idx].getDestination();
		w[idx] = NVE[idx].getWeight();
		//printf("v %d e %d w %d : result %d", v[idx], e[idx], w[idx], NVE[idx].cell);
	}
}


__global__ void mark_differentUV(unsigned int* flag, unsigned int* v, unsigned int* e, unsigned int n) {
	//TODO: consider go with vertex indeces [1, n+1].
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx == 0) {
		//in case of zero it's enough not to be a self loop
		if (v[idx] != e[idx])
			flag[0] = 1;
	}
	else if (idx < n) {
		//we have to pick edges that are not self-loops 
		//and which have different (source, destination)
		if (v[idx] != e[idx] &&
			(v[idx - 1] != v[idx] || e[idx - 1] != e[idx])) {
			flag[idx] = 1;
		}

	}

}

__global__ void mark_differentU(unsigned int* flag, unsigned int* v, unsigned int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < n - 1) {
		if (v[idx] != v[idx + 1]) {
			flag[idx] = 1;
		}

	}

}

struct minus1 : public thrust::unary_function<unsigned int, unsigned int>
{
	__host__ __device__
		unsigned int operator()(unsigned int x)
	{

		return x - 1;
	}
};
struct plus1 : public thrust::unary_function<unsigned int, unsigned int>
{
	__host__ __device__
		unsigned int operator()(unsigned int x)
	{

		return x + 1;
	}
};

struct less_than : public thrust::less<NVEcell>
{
	__host__ __device__
		bool operator()(NVEcell a, NVEcell b)
	{

		return a.cell < b.cell;
	}
};


/// on X in position pointed by edgePtr[src] we can obtain the min couple (weight, dest) for that src vertex.
void minOutgoingEdge(DatastructuresOnGpu* onGPU) {

	fill << < grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->X, onGPU->edges,
		onGPU->numEdges, createMask(0, VERTEX_SIZE));
	//or cudaMemcpy(onGPU->X, onGPU->edges, sizeof(unsigned int)* onGPU->numEdges, cudaMemcpyDeviceToDevice);
	#ifdef SYNCH 
 cudaDeviceSynchronize(); 
 #endif
	fill << < grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->X, onGPU->weights,
		onGPU->numEdges, createMask(VERTEX_SIZE, WEIGHT_SIZE), VERTEX_SIZE);

#ifdef PEDANTIC
	std::cout << "X after fill with weights from bit " << VERTEX_SIZE << " to " << VERTEX_SIZE + WEIGHT_SIZE << "; and edge ids from bit 0 to " << VERTEX_SIZE << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges, VERTEX_SIZE);
#endif

	cudaMemset(onGPU->F, 0, sizeof(unsigned int) * onGPU->numEdges);
	#ifdef SYNCH 
 cudaDeviceSynchronize(); 
 #endif
	mark_edge_ptr << < grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->F, onGPU->edgePtr, onGPU->numVertices);
	#ifdef SYNCH 
 cudaDeviceSynchronize(); 
 #endif
	segmentedMinScanInCuda(onGPU->X, onGPU->X, onGPU->F, onGPU->numEdges);
	#ifdef SYNCH 
 cudaDeviceSynchronize(); 
 #endif
#ifdef PEDANTIC
	std::cout << "Source vertex index in vertex list: " << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
	std::cout << "X after segmented min scan: " << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges, VERTEX_SIZE);
#endif
}


void moveMinWeightsAndSuccessor(DatastructuresOnGpu* onGPU) {
	//we move to S: the couples (W,V) with minimum W. 

	copyIndirected <<< grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >>>(onGPU->F, onGPU->X, onGPU->edgePtr, onGPU->numVertices, onGPU->numEdges);
	#ifdef SYNCH 
 cudaDeviceSynchronize(); 
 #endif

#ifdef PEDANTIC
	std::cout << "====================== Move min weights and successor ====================" << std::endl;
	std::cout << "X after segmented min scan:  (w,v)" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges, VERTEX_SIZE);
	std::cout << "EdgePtr " << std::endl;
	debug_device_ptr(onGPU->edgePtr, onGPU->numVertices);
#endif
#if defined(PEDANTIC) || defined(DEBUG) 
	std::cout << "Min outgoing edge for each node (in F) (w,v)" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numVertices, VERTEX_SIZE);
#endif
	//eliminate cycles in s[s[i]] = i;
	//unpack weights and outgoing edge in X, S.
	// at the same time perform eliminate cycles S[S[i]] = i
	//move successors and weight in S and X respectively.
	// S[S[i]] = i are addressed
	moveWeightsAndSuccessors << < grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->F, onGPU->X, onGPU->S, onGPU->numVertices);
	#ifdef SYNCH 
	cudaDeviceSynchronize(); 
	#endif
}



void computeCosts(DatastructuresOnGpu* onGPU) {
#ifdef PEDANTIC
	std::cout << "weights of min cut:" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numVertices);
#endif
	//use weights stored in X to
	//compute additional cost of this step of min spanning tree.
	//note: moveWeights set to 0 elements S[S[i]] = i;

	thrust::device_ptr<unsigned int> Ws(onGPU->X);
	thrust::inclusive_scan(Ws, Ws + onGPU->numVertices, Ws);

	cudaDeviceSynchronize(); 


#ifdef PEDANTIC
	std::cout << "sum of min weights outgoing edges performed on X (w,v)" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numVertices);
#endif
	//last cell of F contains the cost.
	//Load into a variable.
	unsigned int deltaCosts = 0;
	cudaMemcpy(&deltaCosts, onGPU->X + onGPU->numVertices - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
#if defined(DEBUG ) || defined(PEDANTIC)
	std::cout << "Additional costs" << deltaCosts << std::endl;
#endif
	//update the current MST cost.
	onGPU->cost += deltaCosts;
}

void buildSuccessor(DatastructuresOnGpu* onGPU) {
#ifdef PEDANTIC
	std::cout << "Successor: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
#endif
	//compute successor

	if (sizeof(unsigned int) * onGPU->numVertices < onGPU->maxSharedBytes) {
		int block_dim = std::min(1024, ((int)(onGPU->numVertices + 16) / 17));
		replaceTillFixedPointInShared << <1, block_dim, sizeof(unsigned int) * onGPU->numVertices >> >(onGPU->S, onGPU->numVertices);
	}
	else {
		unsigned int max_vertex_in_shared = onGPU->maxSharedBytes / sizeof(unsigned int);
		replaceTillFixedPoint <<<1, 1024,  max_vertex_in_shared* sizeof(unsigned int) >>>(onGPU->S, onGPU->numVertices, max_vertex_in_shared);
	}
cudaDeviceSynchronize(); 

#ifdef PEDANTIC
	std::cout << "Successor after fixed point: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
#endif
}

void buildSupervertexId(DatastructuresOnGpu* onGPU) {
	thrust::device_ptr<unsigned int>
		s_on_gpu(onGPU->S), v_on_gpu(onGPU->vertices),
		f_on_gpu(onGPU->F);
	//1. aggregate vertices with the same supervertex id
	//   in order to be able to reconstruct which supervertex is related to which vertex
	//	 we move according to the transformation the original position in which is was. 
	//	 The original position will be temporarely stored in v.
#ifdef PEDANTIC
	std::cout << "Rebuild graph representation for the next iteration" << std::endl;
	std::cout << "Source: " << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->numVertices);
	std::cout << "Renaming to supervertex: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
#endif
	fill << < grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->vertices, 1, onGPU->numVertices);
cudaDeviceSynchronize(); 
	//use exclusive scan so that first element get 0 as original position
	thrust::exclusive_scan(v_on_gpu, v_on_gpu + onGPU->numVertices, v_on_gpu);
cudaDeviceSynchronize(); 
	//2. order according to successor and move accordingly the array v
	thrust::sort_by_key(s_on_gpu, s_on_gpu + onGPU->numVertices, v_on_gpu);
	//3. create a flag that contains 1 in position i , when  supervertex[i-1] != supervertex[i] , 0 otherwise.
	cudaMemset(onGPU->F, 0, onGPU->numVertices * sizeof(unsigned int));
	cudaDeviceSynchronize(); 
	mark_discontinuance << < grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->F, onGPU->S, onGPU->numVertices);
	cudaDeviceSynchronize(); 

#ifdef PEDANTIC
	std::cout << "Pick different supervertex" << std::endl;
	std::cout << "vertex index: " << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->numVertices);
	std::cout << "supervertex: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
	std::cout << "mark supervetex end: " << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numVertices);
#endif

	//4. perform a scan to build supervertex identifier.
	thrust::inclusive_scan(f_on_gpu, f_on_gpu + onGPU->numVertices, f_on_gpu);
	cudaDeviceSynchronize(); 

#ifdef PEDANTIC
	std::cout << "id of supervetex: " << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numVertices);
#endif
	//5. once that for each vertex we have determined the new supervetex id.
	//	 Reconstruct a map that permits to move from the vertex->new_supervertex_id. 
	//   The map will be placed in S. 
	//use v that have been reordered to recreate a map v->new_supervertex_id according to the original order.

	thrust::scatter(f_on_gpu, f_on_gpu + onGPU->numVertices, v_on_gpu, s_on_gpu);
	cudaDeviceSynchronize(); 
#ifdef PEDANTIC
	std::cout << "vertex ->id of supervetex: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
#endif
}



void orderUVW(DatastructuresOnGpu* onGPU) {
	thrust::device_ptr<unsigned int>
		s_on_gpu(onGPU->S), v_on_gpu(onGPU->vertices),
		f_on_gpu(onGPU->F), e_on_gpu(onGPU->edges),
		w_on_gpu(onGPU->weights), eptr_on_gpu(onGPU->edgePtr),
		x_on_gpu(onGPU->X);
	thrust::device_ptr<NVEcell> NVE_on_gpu(onGPU->NVE);

	//1. consider the arrays that describe the edges u-w->v.
	//1.a remap u,v with new supervertex id.
	//	v = source
	//	edges = destination
	//	weights = weights
	cudaMemset(onGPU->F, 0, onGPU->numEdges* sizeof(unsigned  int));
	//since elements 0 has to obtain flag = 0 use edgePTR excluding the first cell in the mark_edge_ptr call
	mark_edge_ptr << <grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> > (onGPU->F, onGPU->edgePtr, onGPU->numVertices);
	cudaDeviceSynchronize(); 
#ifdef PEDANTIC
	std::cout << "=============================== reorder according to U,V,W where U-w->V : ===============" << std::endl;
	std::cout << "Edge ptr: " << std::endl;
	debug_device_ptr(onGPU->edgePtr, onGPU->numVertices);
	std::cout << "set 1 at start of the edge segments to build source vertex: " << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
#endif
	//after inclusive scan f[i] = index of source vertex

	thrust::inclusive_scan(f_on_gpu, f_on_gpu + onGPU->numEdges, f_on_gpu);
	cudaDeviceSynchronize(); 
#ifdef PEDANTIC
	std::cout << "source of the edge:" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
	std::cout << "destination of the edge:" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->numEdges);
	std::cout << "Map:" << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
#endif
	//convert edge source & destination thanks to vector S.
	thrust::gather(f_on_gpu, f_on_gpu + onGPU->numEdges, s_on_gpu, v_on_gpu);
	thrust::gather(e_on_gpu, e_on_gpu + onGPU->numEdges, s_on_gpu, x_on_gpu);
	//we would like to store all info in e.
	//mem cpy is necessary due to the fact that gather input /output has to reside on different memory areas
	cudaMemcpy(onGPU->edges, onGPU->X, sizeof(unsigned int)*onGPU->numEdges, cudaMemcpyDeviceToDevice);

#ifdef PEDANTIC
	std::cout << "Transformated source of the edge (Result)  " << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->numEdges);
	std::cout << "Transformated destination of the edge (Result)" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->numEdges);
#endif
	//2. reored triplet u,v,w. But instead of using a single array . 
	//	 Exploit a stable sort and a perform three different order carrying along a index represented by X.
	//   as in radix sort go from least signicants bits to most significative bits,
	//2.a. create index increment.
	fill << < grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->X, 1, onGPU->numEdges);
	cudaDeviceSynchronize(); 
	//2.b. create index by accumulating increments: use exclusive since first element has index 0.
	thrust::exclusive_scan(x_on_gpu, x_on_gpu + onGPU->numEdges, x_on_gpu);
	cudaDeviceSynchronize(); 

	cudaMemset(onGPU->NVE, 0, onGPU->numEdges* sizeof(NVEcell));
	loadNVE << <  grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->NVE, onGPU->vertices, onGPU->edges, onGPU->weights, onGPU->numEdges);

#ifdef PEDANTIC
	std::cout << "Keys(X) before ordering" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
	std::cout << "Source vertices" << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->numEdges);
	std::cout << "Destination vertices" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->numEdges);
	std::cout << "Weights" << std::endl;
	debug_device_ptr(onGPU->weights, onGPU->numEdges);
#endif

	thrust::sort_by_key(NVE_on_gpu, NVE_on_gpu + onGPU->numEdges, x_on_gpu);

	//3.e. last: in order for i that:	f[i] < f[j] and 
	//								edges[i] < edges[j] and
	//								weights[i] < weights[j] 
	// we have to reorder edges and weights arrays according to index x.
	//NOTE: nothing has been said about gather values and output possible overlapping.
	unloadNVE << <  grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->NVE, onGPU->vertices, onGPU->edges, onGPU->weights, onGPU->numEdges);

#ifdef PEDANTIC
	std::cout << "Keys(X) after ordering source, destination, weights" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
	std::cout << "Source vertices" << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->numEdges);
	std::cout << "Destination vertices" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->numEdges);
	std::cout << "Weights" << std::endl;
	debug_device_ptr(onGPU->weights, onGPU->numEdges);
#endif
}

void rebuildEdgeWeights(DatastructuresOnGpu* onGPU) {
	thrust::device_ptr<unsigned int>
		s_on_gpu(onGPU->S), v_on_gpu(onGPU->vertices),
		f_on_gpu(onGPU->F), e_on_gpu(onGPU->edges),
		w_on_gpu(onGPU->weights), eptr_on_gpu(onGPU->edgePtr),
		x_on_gpu(onGPU->X);


	//8. create Edge, weights
	//8.a use F to mark subseqent (v,u) which are not equals and neither self-loops.
	cudaMemset(onGPU->F, 0, onGPU->numEdges* sizeof(unsigned int));
	cudaDeviceSynchronize(); 
	mark_differentUV << <grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->F, onGPU->vertices, onGPU->edges, onGPU->numEdges);
	cudaDeviceSynchronize(); 
	//8.b perform a scan of F to obtain position were to put edge[idx], weight[idx] put the scan result on X
	//	  then use: 
	//		-	x as a map 
	//		-   F as a stencil
	thrust::inclusive_scan(f_on_gpu, f_on_gpu + onGPU->numEdges, x_on_gpu);
	//8.c compute the number of edges available in the next iteration by reading the tail of the scan vector result.

	cudaMemcpy(&onGPU->newNumEdges, onGPU->X + onGPU->numEdges - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//8.d. since index start at 0 subtract 1 from all the indices.
	thrust::transform(x_on_gpu, x_on_gpu + onGPU->numEdges, x_on_gpu, minus1());
	cudaDeviceSynchronize(); 
#ifdef PEDANTIC
	std::cout << "new edges/weights" << std::endl;
	std::cout << "mark relevant edges for edges/weights" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
	std::cout << "positions" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
#endif
	//Since there are multiple elements with the same destination index and we have to insert only those that are in correspondence with a change in x.
	//we exploit a version of scatter that expect a stencil.
	//8.d. move weights
	//since input data and output can't reside on the same memory area we move them temporarily to S
	cudaMemcpy(onGPU->S, onGPU->weights, sizeof(unsigned int)* onGPU->numEdges, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize(); 
	thrust::scatter_if(s_on_gpu, s_on_gpu + onGPU->numEdges, x_on_gpu, f_on_gpu, w_on_gpu);
	//8.e. move destination vertices
	//since input data and output can't reside on the same memory area we mve them temporarily to S
	cudaMemcpy(onGPU->S, onGPU->edges, sizeof(unsigned int)* onGPU->numEdges, cudaMemcpyDeviceToDevice);
	cudaDeviceSynchronize(); 
	thrust::scatter_if(s_on_gpu, s_on_gpu + onGPU->numEdges, x_on_gpu, f_on_gpu, e_on_gpu);
	cudaDeviceSynchronize(); 

#ifdef PEDANTIC
	std::cout << "new edges (" << onGPU->newNumEdges << ") :" << std::endl;
	std::cout << "new weights" << std::endl;
	debug_device_ptr(onGPU->weights, onGPU->newNumEdges);
	std::cout << "new edges" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->newNumEdges);
#endif
}

void rebuildEdgePtr(DatastructuresOnGpu* onGPU) {
	thrust::device_ptr<unsigned int>
		s_on_gpu(onGPU->S), v_on_gpu(onGPU->vertices),
		f_on_gpu(onGPU->F),
		w_on_gpu(onGPU->weights), eptr_on_gpu(onGPU->edgePtr),
		x_on_gpu(onGPU->X);
	//exploit data already computed for Edges in X number of useful edges
	//9. create EdgePTR:
	//9.a find discontinuance in source vertex
	//    since there was a collapsing in the previous step there would always be a edge for each supervertex.
	cudaMemset(onGPU->F, 0, onGPU->numEdges* sizeof(unsigned int));
	cudaDeviceSynchronize(); 
	mark_differentU << <grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->F, onGPU->vertices, onGPU->numEdges);
	cudaDeviceSynchronize(); 
	//9.b perform a inclusive scan on such a flag in order to obtain offset in the edge ptr
	thrust::inclusive_scan(f_on_gpu, f_on_gpu + onGPU->numEdges, s_on_gpu);
	//9.c obtain vertex number by reading from the tail of S
	//and then summing 1
	cudaMemcpy(&onGPU->newNumVertices, onGPU->S + onGPU->numEdges - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize(); 
	onGPU->newNumVertices++;

	//9.d Build edge ptr:
	//when the source vector change pick the number of relevant edges till the previous vertex and then increment by one 
	//9.d.1
	//if on x we have the counted the number of edges that will be preserved
	//   on f we have the stencil that indicate two different consequence sources
	//   on s we have the scan result on the stencil.
#ifdef PEDANTIC
	std::cout << "mark relevant edges for vertices" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
	std::cout << "positions" << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numEdges);
	std::cout << "scan result on relevant edges (already performed at step 7)" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
#endif

	thrust::scatter_if(x_on_gpu, x_on_gpu + onGPU->numEdges, s_on_gpu, f_on_gpu, eptr_on_gpu);
	cudaDeviceSynchronize(); 
	//9.d.2.
	//sum 1 to all to pick up the next
	thrust::transform(eptr_on_gpu, eptr_on_gpu + onGPU->newNumVertices, eptr_on_gpu, plus1());
	cudaDeviceSynchronize(); 
	//set edgeptr of 0 equal to 0.
	cudaMemset(onGPU->edgePtr, 0, sizeof(unsigned int));
	cudaDeviceSynchronize(); 
#ifdef PEDANTIC
	std::cout << "new edge ptr" << std::endl;
	debug_device_ptr(onGPU->edgePtr, onGPU->newNumVertices);
#endif



}

void rebuildVertices(DatastructuresOnGpu* onGPU) {
	thrust::device_ptr<unsigned int>
		v_on_gpu(onGPU->vertices),
		x_on_gpu(onGPU->X);


	fill << < grid(onGPU->newNumVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->X, 1, onGPU->newNumVertices);
	cudaDeviceSynchronize(); 

	thrust::exclusive_scan(x_on_gpu, x_on_gpu + onGPU->newNumVertices, v_on_gpu);
	cudaDeviceSynchronize(); 

#ifdef PEDANTIC
	std::cout << "new vertices (" << onGPU->newNumVertices << ") :" << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->newNumVertices);
#endif



}
