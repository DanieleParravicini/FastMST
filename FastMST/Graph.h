#pragma once

#include <boost\graph\adjacency_list.hpp>
#include <fstream>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DataStructureOnGpu.h"

using namespace boost;

typedef property<edge_weight_t, int> EdgeWeightProperty;
typedef adjacency_list<vecS, vecS, undirectedS, no_property, EdgeWeightProperty> Graph;
typedef Graph::edge_descriptor Edge;
typedef property_map<Graph, edge_weight_t>::type WeightMap;


void loadGraphFromFile(std::string path, Graph& g);
void printForWebgraphviz(Graph &g);
void toGraph(Graph &g, DatastructuresOnGpu* onGPU);