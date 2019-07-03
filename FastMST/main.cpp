#pragma once
#include <iostream>
#include <string>
#include <vector>


#include "mst.h"
#include "Graph.h"
#include "CompactGraph.h"
#include <boost/graph/kruskal_min_spanning_tree.hpp>

void create_cord(int num_of_vertex, Graph& g) {
	//this procedure create a simple test example 
	//with a great number of edges that are all connected in sequence 
	// for all vertex v : v-1 -- 1 --> v .
	//it's easy to compute the MST cost as : num_of_vertex -1;
	int mask_v = createMask(0, VERTEX_SIZE);
	int mask_w = createMask(0, WEIGHT_SIZE);

	for (unsigned int u = 1; u < num_of_vertex; u++) {
		boost::add_edge(u & mask_v, (u - 1) & mask_v, 1 & mask_w, g);
	}
}

int main(int argc, char ** argv) {

	Graph g1;
	//loadGraphFromFile("testGraph/USA-road-d.NY.gr", g1);
	//loadGraphFromFile("testGraph/rome99.gr", g1);
	//loadGraphFromFile("testGraph/es4.3.p136.gr", g1);
	//TODO make a simple file example, taken from foundamentals of operations research.

	create_cord(4096, g1); //--> too easy it takes just 1 iteration.

	printForWebgraphviz(g1);

	struct InSpanning {
		std::set<Edge> edges;
		bool operator()(Edge e) const { return edges.count(e) > 0; }
	} spanning;

	boost::kruskal_minimum_spanning_tree(g1, std::inserter(spanning.edges, spanning.edges.end()));

	int cost = 0;
	for (Edge e : spanning.edges) {
		cost += boost::get(boost::edge_weight, g1, e);
	}

	int cost1 = mst(g1);
	if (cost != cost1) {
		std::cout << "KO: total cost mst kruscal [" << cost << "] != boruvska [" << cost1 << "]" << std::endl;
		return 1;
	}
	else {
		std::cout << "OK! mst cost: " << cost << std::endl;
		return 0;
	}
}
