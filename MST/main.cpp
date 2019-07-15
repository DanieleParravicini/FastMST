#pragma once
#include <iostream>
#include <string>
#include <vector>


#include "mst.h"
#include "Graph.h"
#include "CompactGraph.h"
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <ctime>

int main(int argc, char ** argv) {
	Graph g1;

	if (argc < 2) {
		std::cout << "[HELP] Usage of this program: invoke MST <name of a file containing a graph in DIMACS format>" << std::endl;
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



	int cost = mst(g1);
	
	std::cout << "Cost mst boruvska [" << cost << "]" << std::endl;
	return 0;
	
}