#pragma once
#include <string>
#include "Graph.h"

class CompactGraph
{
private:

public:
	std::vector<unsigned int> vertices;
	std::vector<unsigned int> edgePtr;
	int edgesCount = 0;
	std::vector<unsigned int>edges;
	std::vector<unsigned int>weights;

	CompactGraph(std::vector<std::vector<int>> weightMatrix);
	CompactGraph(Graph &g);
	~CompactGraph();

	void print();
};

