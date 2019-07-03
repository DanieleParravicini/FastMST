#include "Utils.h"



int grid(int n, int block) {
	return (n + block - 1) / block;
}

__global__ void copyIndirected(unsigned int* dst, unsigned int* src, unsigned int * ptr, unsigned int n, unsigned int m) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < n - 1) {
		dst[idx] = src[ptr[idx + 1] - 1];
	}
	else if (idx = n - 1) {
		dst[idx] = src[m - 1];
	}
}

__global__ void moveIndirected(int* dst, int* src, int * ptr, int n) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < n) {
		dst[ptr[idx]] = src[idx];
	}
}

template<int mask, int shift>
int unMaskAndShift(int a);


template< int op(int)>
__global__ void copy(int* src, int* dst, int n) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < n) {
		dst[idx] = op(src[idx]);
	}
}

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


unsigned int* MarkEdgeSegments(DatastructuresOnGpu onGPU) {
	cudaError_t status;
	unsigned int * flags;
	try {
		status = cudaMalloc(&flags, sizeof(unsigned int)*onGPU.numEdges);
		if (status != cudaError::cudaSuccess)
			throw status;
		MarkEdgeSegmentsOnGpu(onGPU, flags);

	}
	catch (...) {
		cudaFree(flags);
		throw status;
	}
	//error
	return flags;
}

void MarkEdgeSegmentsOnGpu(DatastructuresOnGpu onGPU, unsigned int* flags) {
	cudaMemset(flags, 0, onGPU.numEdges* sizeof(unsigned int));
	mark_edge_ptr << <  grid(onGPU.numVertices, BLOCK_SIZE), BLOCK_SIZE >> > (flags, onGPU.edgePtr, onGPU.numVertices);
}


__host__ __device__ unsigned int createMask(int start, int width) {
	unsigned int mask = 1;
	mask = (mask << width) - 1;
	mask = mask << start;
	return mask;
}

void* moveToGpu(void* src, size_t sizeinBytes) {
	void* dst;
	cudaError_t	status;
	try {
		status = cudaMalloc(&dst, sizeinBytes);
		if (status != cudaError::cudaSuccess)
			throw status;
		status = cudaMemcpy(dst, src, sizeinBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
		if (status != cudaError::cudaSuccess)
			throw status;
	}
	catch (...) {
		cudaFree(dst);
		throw status;
	}
	return dst;
}

void* moveToGpu(unsigned int* src, unsigned int size) {
	return moveToGpu((void*)src, sizeof(unsigned int)*size);
}

void* moveToGpu(std::vector<int> src) {
	return moveToGpu((void*)&src[0], sizeof(unsigned int)*src.size());
}

void debug_device_ptr(unsigned int* ptr, unsigned  int items) {

	unsigned int* x;
	x = (unsigned int*)malloc(sizeof(unsigned int)*items);
	cudaMemcpy(x, ptr, sizeof(unsigned int)*items, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < items; i++) {
		std::cout << std::to_string(x[i]) << ";";
	}
	std::cout << "\n";
	free(x);
}

void debug_device_ptr(unsigned int* ptr, unsigned int items, unsigned int nrbit) {

	unsigned int* x;
	x = (unsigned int*)malloc(sizeof(unsigned int)*items);
	cudaMemcpy(x, ptr, sizeof(unsigned int)*items, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	for (unsigned int i = 0; i < items; i++) {
		unsigned int mask = createMask(0, nrbit);
		std::cout << "(" << std::to_string((x[i] & ~mask) >> nrbit) << "," << std::to_string(x[i] & mask) << ")" << ";";
	}
	std::cout << "\n";
	free(x);
}



void segmentedScan(unsigned int* out, unsigned int* in, unsigned int * flags, unsigned int width) {
	unsigned int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(unsigned int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	segmentedScanInCuda(out, in, tmpKeys, width);
	cudaFree(tmpKeys);
}


void segmentedScanInCuda(unsigned int* out, unsigned int* in, unsigned int * flags, unsigned int width)
{
	thrust::device_ptr<unsigned int> dev_in(in);
	thrust::device_ptr<unsigned int> dev_out(out);
	thrust::device_ptr<unsigned int> dev_flags(flags);

	thrust::inclusive_scan(dev_flags, dev_flags + width, dev_flags);
	thrust::exclusive_scan_by_key(dev_flags, dev_flags + width, dev_in, dev_out);
}

struct min_op {
	__host__ __device__
		int  operator()(const int a, const int b) const
	{
		return (a < b) ? a : b;
	}
};

void segmentedMinScan(unsigned int* out, unsigned int* in, unsigned int* flags, unsigned int width) {
	//keys are only temporary can we use shared memory?
	unsigned int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(unsigned int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);

	segmentedMinScanInCuda(out, in, tmpKeys, width);

	cudaFree(tmpKeys);
}

///
/// flags will be touched and invalidated. flags will contain vertex identifier
void segmentedMinScanInCuda(unsigned int* out, unsigned int* in, unsigned int* flags, unsigned int width) {

	thrust::device_ptr<unsigned int> dev_in(in);
	thrust::device_ptr<unsigned int> dev_out(out);
	thrust::device_ptr<unsigned int> dev_flags(flags);

	thrust::inclusive_scan(dev_flags, dev_flags + width, dev_flags);

	thrust::equal_to<unsigned int> binary_pred;
	thrust::minimum <unsigned int> binary_op;

	thrust::inclusive_scan_by_key(dev_flags, dev_flags + width, dev_in, dev_out, binary_pred, binary_op);
}

void split(int* data, int* flags, int width) {
	int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	splitInCuda(data, tmpKeys, width);

	cudaFree(tmpKeys);
}

void splitInCuda(int* data, int* flags, int width) {
	thrust::device_ptr<int> dev_flags(flags);
	thrust::device_ptr<int> dev_data(data);

	thrust::inclusive_scan(dev_flags, dev_flags + width, dev_flags);
	thrust::sort_by_key(dev_flags, dev_flags + width, dev_data);
}

template<typename Tdst, typename Tsrc>
__global__ void fill(Tdst* out, Tsrc* src, int width, int from) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = (src[idx] << from) | (out[idx]);
	}
}

__global__ void fill(unsigned int* out, unsigned int immediate, unsigned int width) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = immediate;
	}
}

__global__ void fill(unsigned int* out, unsigned int* src, unsigned int width, unsigned int mask) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = (src[idx] & mask) | (out[idx] & ~mask);
	}
}

__global__ void fill(unsigned int* out, unsigned int* src, unsigned int width, unsigned int mask, unsigned int from) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = ((src[idx] << from) & mask) | (out[idx] & ~mask);
	}
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

__global__ void mark_differentU(unsigned int* flag, unsigned int* v, unsigned int* e, unsigned int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx == 0) {

		flag[0] = 1;
	}
	else if (idx < n) {
		if (v[idx] != v[idx - 1]) {
			flag[idx] = 1;
		}

	}

}
/// on X in position pointed by edgePtr[src] we can obtain the min couple (weight, dest) for that src vertex.
void minOutgoingEdge(DatastructuresOnGpu* onGPU) {


	fill << < grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->X, onGPU->edges, onGPU->numEdges, createMask(0, VERTEX_SIZE));
#ifdef DEBUG
	std::cout << "X after fill with destination vertices from bit 0 to 22: ";
	debug_device_ptr(onGPU->X, onGPU->numEdges);
#endif

	fill << < grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->X, onGPU->weights, onGPU->numEdges, createMask(VERTEX_SIZE, WEIGHT_SIZE), VERTEX_SIZE);
#ifdef DEBUG
	std::cout << "X after fill with weights from bit 22 to 32: ";
	debug_device_ptr(onGPU->X, onGPU->numEdges, 22);
#endif

	fill << <grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->F, 0, onGPU->numEdges);
	mark_edge_ptr << < grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->F, onGPU->edgePtr, onGPU->numVertices);
	cudaDeviceSynchronize();
#ifdef DEBUG
	{
		std::cout << "F before segmented min scan: ";
		debug_device_ptr(onGPU->F, onGPU->numEdges);
	}
#endif
	//TODO: what about launching a kernel for each vertex, paying in extra thread overhead but permit to mantain distinguished
	// weights and destination vertices 
	segmentedMinScanInCuda(onGPU->X, onGPU->X, onGPU->F, onGPU->numEdges);
#ifdef DEBUG
	{
		std::cout << "F after segmented min scan: ";
		debug_device_ptr(onGPU->F, onGPU->numEdges);
		std::cout << "X after segmented min scan: ";
		debug_device_ptr(onGPU->X, onGPU->numEdges, 22);
	}
#endif


}

__global__ void replaceTillFixedPoint(unsigned int * S, unsigned int n) {
	//consider to move S of block into a shared memory.
	//TDOD:refacytoring
	__shared__ bool flag;
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		unsigned int new_s = S[idx];

		unsigned int old_s;
		do {
			flag = false;


			old_s = new_s;
			new_s = S[new_s];

			if (old_s != new_s)
				flag = true;

			__syncthreads();

			//S[idx] = new_s;

		} while (flag);

		S[idx] = new_s;
		//printf("in the end %d: %d \n", idx, S[idx]);
	}

}
__global__ void map(int*dst, int* src, int* map, int n) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < n) {
		dst[idx] = map[src[idx]];
	}
}

void buildSuccessorAndUpdateCosts(DatastructuresOnGpu* onGPU) {
	//we move to S: the couples (W,V) with minimum W. 

	copyIndirected << < grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->S, onGPU->X, onGPU->edgePtr, onGPU->numVertices, onGPU->numEdges);
	cudaDeviceSynchronize();

#ifdef DEBUG
	std::cout << "origin X (w,v)" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges, VERTEX_SIZE);
	std::cout << "EdgePtr (w,v)" << std::endl;
	debug_device_ptr(onGPU->edgePtr, onGPU->numVertices);
	std::cout << "Min outgoing edge for each node (in S) (w,v)" << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices, VERTEX_SIZE);
#endif
	//eliminate cycles in s[s[i]] = i;
	//unpack weights and outgoing edge in F, S.
	// at the same time perform eliminate cycles S[S[i]] = i
	cudaMemcpy(onGPU->X, onGPU->S, sizeof(unsigned int)* onGPU->numVertices, cudaMemcpyDeviceToDevice);
	//move successors and weight in S and F respectively.
	// S[S[i]] = i are addressed

	moveWeightsAndSuccessors << < grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->X, onGPU->F, onGPU->S, onGPU->numVertices);
#ifdef DEBUG
	std::cout << "Successor: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
	std::cout << "weights of min cut:" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numVertices);
#endif
	//use weights stored in F to
	//compute additional cost of this step of min spanning tree.
	//note: moveWeights set to 0 elements S[S[i]] = i;
	cudaDeviceSynchronize();
	thrust::device_ptr<unsigned int> Ws(onGPU->F);
	thrust::inclusive_scan(Ws, Ws + onGPU->numVertices, Ws);
#ifdef DEBUG
	std::cout << "sum of min weights outgoing edges performed on F (w,v)" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numVertices);
#endif
	//last cell of F contains the cost.
	//Load into a variable.
	unsigned int deltaCosts = 0;
	cudaMemcpy(&deltaCosts, onGPU->F + onGPU->numVertices - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//update the current MST cost.
	onGPU->cost += deltaCosts;

	//compute successor
	replaceTillFixedPoint << <grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->S, onGPU->numVertices);
	cudaDeviceSynchronize();
#ifdef DEBUG
	std::cout << "Successor after fixed point: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
	debug_device_ptr(onGPU->F, onGPU->numVertices);
#endif
}

struct minus1 : public thrust::unary_function<int, int>
{
	__host__ __device__
		int operator()(int x)
	{
		// note that using printf in a __device__ function requires
		// code compiled for a GPU with compute capability 2.0 or
		// higher (nvcc --arch=sm_20)
		return x - 1;
	}
};

void rebuildConsedGraphrepresentation(DatastructuresOnGpu *onGPU) {
	thrust::device_ptr<unsigned int>
		s_on_gpu(onGPU->S), v_on_gpu(onGPU->vertices),
		f_on_gpu(onGPU->F), e_on_gpu(onGPU->edges),
		w_on_gpu(onGPU->weights), eptr_on_gpu(onGPU->edgePtr),
		x_on_gpu(onGPU->X), c_on_gpu(onGPU->C);


	//1. aggregate vertices with the same supervertex id
	//2. in order to be able to reconstruct which supervertex is related to which vertex
	//	 we bring an additional value that is the vertex.
#ifdef DEBUG
	std::cout << "Rebuild graph representation for the next iteration" << std::endl;
	std::cout << "Source: " << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->numVertices);
	std::cout << "Renaming to supervertex: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
#endif
	thrust::sort_by_key(s_on_gpu, s_on_gpu + onGPU->numVertices, v_on_gpu);
	//3. create a flag that contains 1, when  supervertex[i] != supervertex[i+1] , 0 otherwise.
	cudaMemset(onGPU->F, 0, onGPU->numVertices * sizeof(unsigned int));
	mark_discontinuance << <grid(onGPU->numVertices, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->F, onGPU->S, onGPU->numVertices);
#ifdef DEBUG
	std::cout << "Pick different supervertex" << std::endl;
	std::cout << "vertex: " << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->numVertices);
	std::cout << "supervertex: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
	std::cout << "mark supervetex end: " << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numVertices);
#endif
	//4. perform a scan to build supervertex identifier.
	thrust::inclusive_scan(f_on_gpu, f_on_gpu + onGPU->numVertices, f_on_gpu);
#ifdef DEBUG
	std::cout << "id of supervetex: " << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numVertices);
#endif
	//5. once that for each vertex we have determined the new supervetex id.
	//	 build a map curr_vertex->new_supervertex_id. this map is placed in S. 
	//moveIndirected << <grid(onGpu->numVertices, BLOCK_SIZE), BLOCK_SIZE >>> (onGpu->S, onGpu->F, onGpu->vertices, onGpu->numVertices);
	//use v that have been reordered to recreate a map v->new_supervertex_id this information is stored on S.
	thrust::scatter(f_on_gpu, f_on_gpu + onGPU->numVertices, v_on_gpu, s_on_gpu);
#ifdef DEBUG
	std::cout << "vertex ->id of supervetex: " << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
#endif
	//6. consider the arrays that describe the edges u-w->v.
	//6.a remap u,v with new supervertex id.
	//	F = source
	//	edges = destination
	//	weights = weights
	cudaMemset(onGPU->F, 0, onGPU->numEdges* sizeof(unsigned  int));
	mark_edge_ptr << <grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> > (onGPU->F, onGPU->edgePtr, onGPU->numVertices);
#ifdef DEBUG
	std::cout << "Edge ptr: " << std::endl;
	debug_device_ptr(onGPU->edgePtr, onGPU->numVertices);
	std::cout << "set 1 at each end of the edge segments " << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
#endif
	//after inclusive f[i] = source_vertexid
	thrust::inclusive_scan(f_on_gpu, f_on_gpu + onGPU->numEdges, f_on_gpu);
#ifdef DEBUG
	std::cout << "Rebuild identifier of source vertex" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
#endif
	//map <<<grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >>> (onGPU->F		, onGPU->F		, onGPU->S, onGPU->numEdges );
	//map <<<grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >>> (onGPU->edges	, onGPU->edges	, onGPU->S, onGPU->numEdges );
#ifdef DEBUG
	std::cout << "Transform source of the edge:" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
	std::cout << "Transform destination of the edge:" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->numEdges);
	std::cout << "Map:" << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numVertices);
#endif
	thrust::gather(f_on_gpu, f_on_gpu + onGPU->numEdges, s_on_gpu, x_on_gpu);
	//mem cpy is necessary due to the fact that gather input /output has to reside on different memory areas
	cudaMemcpy(onGPU->F, onGPU->X, sizeof(unsigned int)*onGPU->numEdges, cudaMemcpyDeviceToDevice);
	thrust::gather(e_on_gpu, e_on_gpu + onGPU->numEdges, s_on_gpu, x_on_gpu);
	cudaMemcpy(onGPU->edges, onGPU->X, sizeof(unsigned int)*onGPU->numEdges, cudaMemcpyDeviceToDevice);
#ifdef DEBUG
	std::cout << "Result of transform source of the edge:" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
	std::cout << "REsult of transform destination of the edge:" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->numEdges);
#endif
	//7. reored triplet u,v,w. But instead of using a single array . 
	//	 Exploit a stable sort and a perform three different order carrying along a index represented by X.
	//   as in radix sort go from least signicants bits to most significative bits,
	//7.a. create index.
	fill << < grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->X, 1, onGPU->numEdges);
	//use exclusive since first eleement has index 0.
	thrust::exclusive_scan(x_on_gpu, x_on_gpu + onGPU->numEdges, x_on_gpu);
#ifdef DEBUG
	std::cout << "Keys(X) before ordering" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
	std::cout << "Weights" << std::endl;
	debug_device_ptr(onGPU->weights, onGPU->numEdges);
	std::cout << "Destination vertices" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->numEdges);
	std::cout << "Source vertices" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);

#endif
	//7.b. stable_sort w.r.t. weights and carry along the modification on indeces
	//NOTE: exploit S as a temporary buffer in order to not corrupt the original array. this dose not hold for the last order.
	cudaMemcpy(onGPU->S, onGPU->weights, sizeof(unsigned int)*onGPU->numEdges, cudaMemcpyDeviceToDevice);
	thrust::stable_sort_by_key(s_on_gpu, s_on_gpu + onGPU->numEdges, x_on_gpu);
	//7.c. before ordering w.r.t. destination vertices first apply reordering according to the index.
	thrust::gather(x_on_gpu, x_on_gpu + onGPU->numEdges, e_on_gpu, s_on_gpu);
	thrust::stable_sort_by_key(s_on_gpu, s_on_gpu + onGPU->numEdges, x_on_gpu);
	//7.d. once again before ordering w.r.t. source vertices first apply reordering according to the index.
	thrust::gather(x_on_gpu, x_on_gpu + onGPU->numEdges, f_on_gpu, f_on_gpu);
	thrust::stable_sort_by_key(f_on_gpu, f_on_gpu + onGPU->numEdges, x_on_gpu);
	//7.e. last: in order for i that:	f[i] < f[j] and 
	//								edges[i] < edges[j] and
	//								weights[i] < weights[j] 
	// we have to reorder edges and weights arrays according to index x.
	thrust::gather(x_on_gpu, x_on_gpu + onGPU->numEdges, e_on_gpu, e_on_gpu);
	thrust::gather(x_on_gpu, x_on_gpu + onGPU->numEdges, w_on_gpu, w_on_gpu);
#ifdef DEBUG
	std::cout << "Keys(X) after ordering source, destination, weights" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
	std::cout << "Weights" << std::endl;
	debug_device_ptr(onGPU->weights, onGPU->numEdges);
	std::cout << "Destination vertices" << std::endl;
	debug_device_ptr(onGPU->edges, onGPU->numEdges);
	std::cout << "Source vertices" << std::endl;
	debug_device_ptr(onGPU->F, onGPU->numEdges);
#endif


	//8. create Edge, weights
	//8.a use X to mark subseqent (v,u) which are not equals and neither self-loops.
	cudaMemset(onGPU->S, 0, onGPU->numEdges* sizeof(unsigned int));
	mark_differentUV << <grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->S, onGPU->F, onGPU->edges, onGPU->numEdges);
	//8.b perform a scan of S to obtain position were to put edge[idx], weight[idx] put the scan result on X
	//	  then use: 
	//		-	x as a map 
	//		-   s as a stencil

	thrust::inclusive_scan(s_on_gpu, s_on_gpu + onGPU->numEdges, x_on_gpu);
	//8.c compute the number of edges available in the next iteration by reading from the tail of the scan vector result.
	unsigned int numEdges;
	cudaMemcpy(&numEdges, onGPU->X + onGPU->numEdges - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	thrust::transform(x_on_gpu, x_on_gpu + onGPU->numEdges, x_on_gpu, minus1());
#ifdef DEBUG
	std::cout << "new edges/weights" << std::endl;
	std::cout << "mark relevant edges for edges/weights" << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numEdges);
	std::cout << "positions" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
#endif
	//Since there are multiple elements with the same destination index and we have to insert only those that are in correspondence with a change in x.
	//we exploit a version of scatter that expect a stencil.
	//8.d. move weights
	//since input data and output can't reside on the same memory area we mve them temporarily on C
	cudaMemcpy(onGPU->C, onGPU->weights, sizeof(unsigned int)* onGPU->numEdges, cudaMemcpyDeviceToDevice);
	thrust::scatter_if(c_on_gpu, c_on_gpu + onGPU->numEdges, x_on_gpu, s_on_gpu, w_on_gpu);
	//8.e. move destination vertices
	//since input data and output can't reside on the same memory area we mve them temporarily on C
	cudaMemcpy(onGPU->C, onGPU->edges, sizeof(unsigned int)* onGPU->numEdges, cudaMemcpyDeviceToDevice);
	thrust::scatter_if(c_on_gpu, c_on_gpu + onGPU->numEdges, x_on_gpu, s_on_gpu, e_on_gpu);

#ifdef DEBUG
	std::cout << "new edges (" << numEdges << ") :" << std::endl;
	std::cout << "new weights" << std::endl;
	debug_device_ptr(onGPU->weights, numEdges);
	std::cout << "new edges" << std::endl;
	debug_device_ptr(onGPU->edges, numEdges);
#endif

	//9. create EdgePTR:
	//9.a find discontinuance in source vertex
	//    since there was a collapsing in the previous step there would always be a edge for each supervertex.
	cudaMemset(onGPU->S, 0, onGPU->numEdges* sizeof(unsigned int));
	mark_differentU << <grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->S, onGPU->F, onGPU->edges, onGPU->numEdges);
	//9.b perform a inclusive scan on such a flag in order to obtain offset in the vertex array
	thrust::inclusive_scan(s_on_gpu, s_on_gpu + onGPU->numEdges, c_on_gpu);
	//9.c obtain vertex number
	unsigned int newNumVertices;
	cudaMemcpy(&newNumVertices, onGPU->C + onGPU->numEdges - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//9.d rescale of 1 unit since flag[0] = 1 but its position has to be 0
	thrust::transform(c_on_gpu, c_on_gpu + onGPU->numEdges, c_on_gpu, minus1());
	//9.e Build edge ptr:
	//if on x we have the counted the number of edges that will be preserved
	//   on s we have the stencil that indicate two different sources
	//   on c we have the scan result on the stencil.
#ifdef DEBUG
	std::cout << "mark relevant edges for vertices" << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numEdges);
	std::cout << "positions" << std::endl;
	debug_device_ptr(onGPU->C, onGPU->numEdges);
	std::cout << "scan result on relevant edges (already performed at step 7)" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
#endif
	thrust::scatter_if(x_on_gpu, x_on_gpu + onGPU->numEdges, c_on_gpu, s_on_gpu, eptr_on_gpu);
	cudaMemset(onGPU->edgePtr, 0, sizeof(unsigned int));
#ifdef DEBUG
	std::cout << "new edge ptr" << std::endl;
	debug_device_ptr(onGPU->edgePtr, newNumVertices);
#endif



	//10. create vertex vector
	//10.a find discontinuance in source vertex
	//    since there was a collapsing in the previous step there would always be a edge for each supervertex.
	//10.b perform a inclusive scan on such a flag in order to obtain offset in the vertex array
	//10.c obtain vertex number
	//10.d rescale of 1 unit since flag[0] = 1 but its position has to be 0
	//already performed by previous steps.
	// S will mark discontinuance
	// C will contain the scan result of S
	// F will contain vertex source.
	//10.e move vertex: only elements identified by a stencil =1 are moved according to the offset already computed
#ifdef DEBUG
	std::cout << "new vertices :" << std::endl;
	std::cout << "relevant edges" << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numEdges);
	std::cout << "position" << std::endl;
	debug_device_ptr(onGPU->X, onGPU->numEdges);
#endif
	thrust::gather_if(x_on_gpu, x_on_gpu + onGPU->numEdges, f_on_gpu, s_on_gpu, v_on_gpu);
#ifdef DEBUG
	std::cout << "new vertices (" << newNumVertices << ") :" << std::endl;
	debug_device_ptr(onGPU->vertices, newNumVertices);
#endif


	//10. Build edge ptr:
	//10.a  place a 1 when we encounter 2 different sources
	//		make an inclusive scan to compute map
	//		x will contain data (number of edges correctly computed before)
	//		vertices array will contain a stencil marking discontinuance between equal sources based on F
	//		s will still contain the map
	//mark_discontinuance << <grid(onGPU->numEdges, BLOCK_SIZE), BLOCK_SIZE >> >(onGPU->vertices, onGPU->F, onGPU->numEdges); substitued by line 548
	/*thrust::inclusive_scan(v_on_gpu, v_on_gpu + onGPU->numEdges, s_on_gpu);
	int numVertices;
	cudaMemcpy(&numVertices, onGPU->S + onGPU->numEdges - 1, sizeof(int), cudaMemcpyDeviceToHost);
	thrust::transform(s_on_gpu, s_on_gpu + onGPU->numEdges, s_on_gpu, minus1());
	#ifdef DEBUG
	std::cout << "mark relevant edges for vertices" << std::endl;
	debug_device_ptr(onGPU->vertices, onGPU->numEdges);
	std::cout << "positions" << std::endl;
	debug_device_ptr(onGPU->S, onGPU->numEdges);
	#endif
	thrust::scatter_if(x_on_gpu, x_on_gpu + onGPU->numEdges, s_on_gpu, v_on_gpu, eptr_on_gpu);
	#ifdef DEBUG
	std::cout << "new edge ptr" << std::endl;
	debug_device_ptr(onGPU->edgePtr, onGPU->numEdges);
	#endif*/

	onGPU->numVertices = newNumVertices;
	onGPU->numEdges = numEdges;

}