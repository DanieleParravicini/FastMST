#pragma once
#include <iostream>
#include <string>
#include <vector>


#include "mst.h"
#include "Graph.h"
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <ctime>

int main(int argc, char ** argv) {
	Graph g1;

	if (argc < 2) {
		std::cout << "[HELP] Usage of this program: invoke verifyMST <name of a file containing a graph in DIMACS format>" << std::endl;
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

	std::set<Edge> edges_golden_model;
	std::vector<Edge> edges_mst;

	clock_t begin = clock();

	boost::kruskal_minimum_spanning_tree(g1, std::inserter(edges_golden_model, edges_golden_model.end()));

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "Time cpu occupied: " << elapsed_secs << " [s]" << std::endl;

	long long int cost = 0;
	for (Edge e : edges_golden_model) {
		cost += boost::get(boost::edge_weight, g1, e);
	}


	long long int cost1 = verifyMst(g1, edges_mst);
	long long int cost2 = 0;
	for (Edge e : edges_mst) {
		cost2 += boost::get(boost::edge_weight, g1, e);
	}
	assert(cost2 == cost1);

	if (cost1 != cost) {
		std::cout << "KO: total cost mst kruscal [" << cost << "] != boruvska [" << cost1 << "]" << std::endl;
		return -1;
	}
	else {
		std::cout << "OK! mst cost: " << cost << std::endl;
		return 0;
	}
}