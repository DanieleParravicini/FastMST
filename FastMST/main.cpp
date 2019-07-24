#pragma once
#include <iostream>
#include <string>
#include <vector>


#include "mst.h"
#include "Graph.h"
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <ctime>



void create_cord(unsigned int num_of_vertex, Graph& g) {
	//this procedure create a simple test example 
	//with a great number of edges that are all connected in sequence 
	// for all vertex v : v-1 -- 1 --> v .
	//it's easy to compute the MST cost as : num_of_vertex -1;
	unsigned int mask_v = createMask(0, VERTEX_SIZE);
	unsigned int mask_w = createMask(0, WEIGHT_SIZE);

	for (unsigned int u = 1; u < num_of_vertex; u++) {
		boost::add_edge(u & mask_v, (u - 1) & mask_v, 1 & mask_w, g);
	}
}

int create_cord_n_iterations(unsigned int num_iterations,  Graph& g, int last_used = 0) {
	//this procedure create a simple test example 
	//with a great number of edges that are all connected in sequence 
	if (num_iterations <= 0)
		return last_used;
	
	unsigned int mask_v = createMask(0, VERTEX_SIZE);
	unsigned int mask_w = createMask(0, WEIGHT_SIZE);
	unsigned int i = last_used;

	if (num_iterations > 1) {
		i = create_cord_n_iterations(num_iterations - 1, g, last_used);
	}
	boost::add_edge(i & mask_v, (i + 1) & mask_v,  num_iterations & mask_w, g);
	i++;
	if (num_iterations > 1) {
		i = create_cord_n_iterations(num_iterations - 1, g, i);
	}
	// return last used cell
	return i;
}

int main(int argc, char ** argv) {

	
	Graph g1;

	if (argc < 2) {
		std::cout << "[HELP] Usage of this program: invoke MST.exe <name of a file containing a graph in DIMACS format>" << std::endl;
		return -1;
	}
	loadGraphFromFile(argv[1], g1);
	//loadGraphFromFile("testGraph/USA-road-d.NY.gr", g1);
	//loadGraphFromFile("testGraph/USA-road-d.NY.gr", g1);
	//loadGraphFromFile("testGraph/rome99.gr", g1);
	//loadGraphFromFile("testGraph/es4.3.p136.gr", g1);
	//create_cord(4096, g1); //--> too easy it takes just 1 iteration.
	//create_cord_n_iterations(9, g1); //--> 1024 nodes
	//create_cord_n_iterations(12, g1);



	//printForWebgraphviz(g1);


	std::vector<Edge> edges_mst;
	long long int cost1 = mst(g1, edges_mst);

	long long int cost2 = 0;
	for (Edge e : edges_mst) {
		cost2 += boost::get(boost::edge_weight, g1, e);
	}
	assert(cost2 == cost1);

	std::cout << "Cost mst boruvska [" << cost1 << "]" << std::endl;
	return 0;
	
}

