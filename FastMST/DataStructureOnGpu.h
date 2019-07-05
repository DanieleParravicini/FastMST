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
	unsigned int newNumEdges;
	unsigned int newNumVertices;

	unsigned int* X;
	unsigned int* F;
	unsigned int* S;


	unsigned int cost = 0;

	void printForWebgraphvizrint() {
		unsigned int	*e		= (unsigned int*)malloc(sizeof(unsigned int)* numEdges),
						*w		= (unsigned int*)malloc(sizeof(unsigned int)* numEdges),
						*e_ptr	= (unsigned int*)malloc(sizeof(unsigned int)* numVertices);

		cudaMemcpy(e	, edges		, sizeof(unsigned int) * numEdges, cudaMemcpyDeviceToHost);
		cudaMemcpy(w	, weights	, sizeof(unsigned int) * numEdges, cudaMemcpyDeviceToHost);
		cudaMemcpy(e_ptr, edgePtr	, sizeof(unsigned int) * numVertices, cudaMemcpyDeviceToHost);

		std::cout << "Graph {" << std::endl;
		//maybe considering first fill vertices array
		//and then search in it. to obtain the identifier in our datastructure. 
		
		for (unsigned int i = 0; i < numVertices-1; i++) {
			for (unsigned int v = e_ptr[i]; v < e_ptr[i + 1]; v++ ) {
				std::cout << i << " -- " << e[v] << "[ label=\"" << w[v] << "\"]" << std::endl;
			}
		}
		for (unsigned int v = e_ptr[numVertices - 1]; v < numEdges; v++) {
			std::cout << numVertices-1 << " -- " << e[v] << "[ label=\"" << w[v] << "\"]" << std::endl;
		}

		std::cout << std::endl << "}" << std::endl;

		free(e);
		free(w);
		free(e_ptr);
	}

};