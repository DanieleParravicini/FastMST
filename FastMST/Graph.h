#pragma once

#include <boost\graph\adjacency_list.hpp>
//#include <boost\graph\adjacency_matrix.hpp>
#include <fstream>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DataStructureOnGpu.h"

#include <boost/graph/rmat_graph_generator.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>

using namespace boost;

typedef property<edge_weight_t, int> EdgeWeightProperty;
typedef adjacency_list<vecS, vecS, undirectedS, no_property, EdgeWeightProperty, no_property> Graph;

typedef Graph::edge_descriptor Edge;
typedef property_map<Graph, edge_weight_t>::type WeightMap;

typedef boost::rmat_iterator<boost::minstd_rand, Graph> RMATGen;
typedef boost::erdos_renyi_iterator<boost::minstd_rand, Graph> ERGen;


void loadGraphFromFile(std::string path, Graph& g);
void writeGraphToFile(Graph& g, std::ostream& out);

void generateRandom(int nr_vertices, int nr_edges, Graph& g);
void printForWebgraphviz(Graph &g);
void toGraph(Graph &g, DatastructuresOnGpu* onGPU);