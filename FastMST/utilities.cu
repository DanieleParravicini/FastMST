#include "utilities.cuh"

int createMask(int start, int width) {
	int mask = 1;
	mask = (mask << (width + 1)) - 1;
	mask = mask << start;
	return mask;
}

void* moveToGpu(void* src, int sizeinBytes) {
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

void* moveToGpu(int* src, int size) {
	return moveToGpu((void*)src, sizeof(int)*size);
}

void* moveToGpu(std::vector<int> src) {
	return moveToGpu(&src[0], sizeof(int)*src.size());
}


void segmentedScan(int* out, int* in, int * flags, int width) {
	int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	segmentedScanInCuda(out, in, tmpKeys, width);
}


void segmentedScanInCuda(int* out, int* in, int * flags, int width)
{

}
struct min_op {
	__host__ __device__
		int  operator()(const int a, const int b) const
	{
		return a < b ? a : b;
	}
};

void segmentedMinScan(int* out, int* in, int* flags, int width) {
	//keys are only temporary can we use shared memory?
	int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	segmentedMinScanInCuda(out, in, tmpKeys, width);

	cudaFree(tmpKeys);
}

///
/// flags will be touched and invalidated. flags will contain vertex identifier
void segmentedMinScanInCuda(int* out, int* in, int* flags, int width) {

	thrust::inclusive_scan(flags, flags + width, flags);

	thrust::equal_to<int> binary_pred;
	min_op binary_op;

	thrust::inclusive_scan_by_key(flags, flags + width, in, out, binary_pred, binary_op);
}

void split(int* data, int* flags, int width) {
	int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	splitInCuda(data, flags, width);

	cudaFree(tmpKeys);
}

void splitInCuda(int* data, int* flags, int width) {
	thrust::inclusive_scan(flags, flags + width, flags);
	thrust::sort_by_key(flags, flags + width, data);
}

__global__ void fill(int* out, int immediate, int width) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = immediate;
	}
}

__global__ void fill(int* out, int* src, int width, int mask) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = (src[idx] & mask) | (out[idx] & !mask);
	}
}

__global__ void fill(int* out, int* src, int width, int mask, int from) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = ((src[idx] << from) & mask) | (out[idx] & !mask);
	}
}

__global__ void mark_edge_ptr(int* out, int* ptr, int width) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[ptr[idx]] = 1;
	}
}


__global__ void ScanKernel(int* out, int *in, int num) {
	__shared__ int* temp;

}

template<class OP, class T> 
__device__ T scan_warp(volatile T *ptr, const unsigned int idx) {
	const unsigned int lane = idx & 31; // index of thread in warp (0..31)
	if (lane >= 1) ptr[idx] = OP::apply(ptr[idx - ­? 1], ptr[idx]); 
	if (lane >= 2) ptr[idx] = OP::apply(ptr[idx - ­? 2], ptr[idx]); 
	if (lane >= 4) ptr[idx] = OP::apply(ptr[idx - ­? 4], ptr[idx]); 
	if (lane >= 8) ptr[idx] = OP::apply(ptr[idx - ­? 8], ptr[idx]); 
	if (lane >= 16) ptr[idx] = OP::apply(ptr[idx - ­? 16], ptr[idx]);
	return (lane>0) ? ptr[idx - ­?1] : OP::identity();
}

template<class OP, class T> 
__device__ void scan_block(volatile T *ptr, const unsigned int idx) {
	const unsigned int lane = idx & 31; // index of thread in warp (0..31) const unsigned int warpid = idx >> 5;
	T val = scan_warp<OP, T>(ptr, idx); 
	if (lane == 31) 
		ptr[warpid] = ptr[idx];

	__syncthreads();
	if (warpid == 0) 
		scan_warp<OP, T>(ptr, idx); // Step 3. scan to accumulate bases __syncthreads();
	if (warpid > 0)// Step 4. apply bases to all elements
		val = OP::apply(ptr[warpid - ­?1], val);
	
	__syncthreads();
	ptr[idx] = val;
}  
							  // Step 1. per-­?warp partial scan // Step 2. copy partial-­?scan bases
__global__ void getMinNodes() {

}