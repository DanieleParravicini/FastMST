#include "utilities.cuh"

int createMask(int start, int width) {
	int mask = 1;
	mask = (mask << width) - 1;
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

std::string debug_device_ptr(int* ptr, int items) {
	std::string str;
	int* x;
	x = (int*)malloc(sizeof(int)*items);
	cudaMemcpy(x, ptr, sizeof(int)*items, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	for (int i = 0; i < items; i++) {
		str.append(std::to_string(x[i]) + "\t");
	}
	str.append("\n");
	free(x);
	return str;
}


void segmentedScan(int* out, int* in, int * flags, int width) {
	int * tmpKeys;
	cudaMalloc(&tmpKeys, sizeof(int)*width);
	cudaMemcpy(tmpKeys, flags, width, cudaMemcpyKind::cudaMemcpyHostToDevice);
	segmentedScanInCuda(out, in, tmpKeys, width);
	cudaFree(tmpKeys);
}


void segmentedScanInCuda(int* out, int* in, int * flags, int width)
{
	thrust::device_ptr<int> dev_in(in);
	thrust::device_ptr<int> dev_out(out);
	thrust::device_ptr<int> dev_flags(flags);

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

	thrust::device_ptr<int> dev_in(in);
	thrust::device_ptr<int> dev_out(out);
	thrust::device_ptr<int> dev_flags(flags);

	thrust::inclusive_scan(dev_flags, dev_flags + width, dev_flags);

	thrust::equal_to<int> binary_pred;
	thrust::minimum<int> binary_op;

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

__global__ void fill(int* out, int immediate, int width) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = immediate;
	}
}

__global__ void fill(int* out, int* src, int width, int mask) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = (src[idx] & mask) | (out[idx] & ~mask);
	}
}

__global__ void fill(int* out, int* src, int width, int mask, int from) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[idx] = ((src[idx] << from) & mask) | (out[idx] & ~mask);
	}
}

__global__ void mark_edge_ptr(int* out, int* ptr, int width) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < width) {
		out[ptr[idx]] = 1;
	}
}

