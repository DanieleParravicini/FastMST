#pragma once
#include <string>
#include "Graph.h"

class ExpandedGraph
{
private:

public:
	std::vector<unsigned int> vertices;
	std::vector<unsigned int> edgePtr;
	int edgesCount = 0;
	std::vector<unsigned int>edges;
	std::vector<unsigned int>weights;
	std::vector<unsigned int>edgesIds;

	ExpandedGraph(std::vector<std::vector<int>> weightMatrix);
	ExpandedGraph(Graph &g);
	~ExpandedGraph();

	void print();
};

