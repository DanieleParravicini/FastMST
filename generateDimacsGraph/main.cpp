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
	if (argc < 3) {
		std::cout << "[HELP] Usage of this program: invoke generateDimacsGraph.exe <name of file to be generated> <number of vertices> <number of edges>." << std::endl
				  << "Pay attention <number of edges> is only a lower bound." << std::endl
				  << "Pay attention <number of edges> has to be greater than <number of vertices> ";
		return -1;
	}
	Graph g1;

	//loadGraphFromFile("testGraph/USA-road-d.NY.gr", g1);
	//loadGraphFromFile("testGraph/rome99.gr", g1);
	//loadGraphFromFile("testGraph/es4.3.p136.gr", g1);
	//create_cord(4096, g1); //--> too easy it takes just 1 iteration.
	//create_cord_n_iterations(9, g1); //--> 1024 nodes
	//create_cord_n_iterations(12, g1);

	int num_vertices = std::atoi(argv[1]);
	int num_edges = std::max(std::atoi(argv[2]), num_vertices);

	generateRandom(num_vertices, num_edges, g1);
	
	//printForWebgraphviz(g1);

	std::ofstream aFile;
	aFile.open( "generated("+ std::to_string(num_vertices) + ","+ std::to_string(num_edges)+").gr", std::ios::out);
	
	writeGraphToFile(g1, aFile);


}

