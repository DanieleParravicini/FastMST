#include "Utils.h"



int grid(int n, int block) {
	return (n + block - 1) / block;
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

void* moveToGpu(std::vector<unsigned int> src) {
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
		std::cout << "<" << std::to_string((x[i] & ~mask) >> nrbit) << "," << std::to_string(x[i] & mask) << ">" << ";";
	}
	std::cout << "\n";
	free(x);
}


void segmentedMinScan(unsigned int* out, unsigned int* in, unsigned int* flags, unsigned int width) {
	//this function is only a stub for testing purposes
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


__global__ void copyIndirected(unsigned int* dst, unsigned int* src, unsigned int * ptr, unsigned int n, unsigned int m) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < n - 1) {
		dst[idx] = src[ptr[idx + 1] - 1];
	}
	else if (idx == n - 1) {
		dst[idx] = src[m - 1];
	}
}





