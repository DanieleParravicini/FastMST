#include "UVcell.cuh"



__host__ __device__ void UVcell::setDestination(unsigned int d) {
	UW |= d;
}

__host__ __device__ void UVcell::setWeight(unsigned int w) {
	UW |= (w << VERTEX_SIZE);
}

__host__ __device__ void UVcell::setID(unsigned int id) {
	ID = id;
}

__host__ __device__ unsigned int UVcell::getDestination() {
	unsigned int tmp = UW;
	unsigned int mask = createMask(0, VERTEX_SIZE);

	return tmp & mask;
}



__host__ __device__ unsigned int UVcell::getWeight() {
	unsigned int tmp = UW;
	return tmp >> VERTEX_SIZE;
}

__host__ __device__ unsigned int UVcell::getID() {
	return ID;
}



