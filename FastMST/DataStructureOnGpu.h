#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct DatastructuresOnGpu {
	unsigned int* vertices = 0;
	unsigned int* edgePtr = 0;
	unsigned int* weights = 0;
	unsigned int* edges = 0;
	unsigned int numEdges;
	unsigned int numVertices;

	unsigned int* X;
	unsigned int* F;
	unsigned int* S;
	unsigned int* C;

	unsigned int cost = 0;

	void printForWebgraphvizrint() {
		unsigned int	* e		= (unsigned int*)malloc(sizeof(unsigned int)* numEdges),
						*w		= (unsigned int*)malloc(sizeof(unsigned int)* numEdges),
						*e_ptr	= (unsigned int*)malloc(sizeof(unsigned int)* numVertices);

		cudaMemcpy(e	, edges		, sizeof(unsigned int) * numEdges, cudaMemcpyDeviceToHost);
		cudaMemcpy(w	, weights	, sizeof(unsigned int) * numEdges, cudaMemcpyDeviceToHost);
		cudaMemcpy(e_ptr, edgePtr	, sizeof(unsigned int) * numVertices, cudaMemcpyDeviceToHost);

		std::cout << "Graph {" << std::endl;
		//maybe considering first fill vertices array
		//and then search in it. to obtain the identifier in our datastructure. 
		unsigned int i = 0;
		for (unsigned int v = 0; v < numVertices; v++) {
			while (i < numEdges && !(v < numVertices - 1 && i == e_ptr[v + 1])) {

				if (v < e[i]) {
					std::cout << v << " -- " << e[i] << "[ label=\"" << w[i] << "\"]" << std::endl;
				}
				i++;
			}
		}
		std::cout << std::endl << "}" << std::endl;
		free(e);
		free(w);
		free(e_ptr);
	}

};