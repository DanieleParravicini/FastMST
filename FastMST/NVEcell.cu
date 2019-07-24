#include "NVEcell.cuh"


__host__ __device__ bool operator < (const NVEcell &lhs, const NVEcell &rhs)
{
	return (lhs.cell < rhs.cell);
};

__host__ __device__ void NVEcell::setSource(unsigned int s) {
	long long tmp = s;
	tmp <<= WEIGHT_SIZE + VERTEX_SIZE;
	cell |= tmp;
}

__host__ __device__ void NVEcell::setDestination(unsigned int d) {
	long long tmp = d;
	tmp <<= WEIGHT_SIZE;
	cell |= tmp;
}

__host__ __device__ void NVEcell::setWeight(unsigned int w) {
	cell |= w;
}

__host__ __device__ unsigned int NVEcell::getSource() {
	long long tmp = cell;
	tmp >>= (WEIGHT_SIZE + VERTEX_SIZE);

	return (unsigned int)tmp;
}

__host__ __device__ unsigned int NVEcell::getDestination() {
	long long tmp = cell;
	int mask = createMask(0, VERTEX_SIZE);
	tmp >>= WEIGHT_SIZE;

	return (unsigned int)tmp & mask;
}

__host__ __device__ unsigned int NVEcell::getWeight() {
	unsigned int tmp = (unsigned int) cell;
	int mask = createMask(0, WEIGHT_SIZE);

	return tmp & mask;
}
