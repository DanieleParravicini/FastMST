#pragma once
#include <vector>
#include <string>
#include "CompactGraph.h"

class Graph
{
private:
	std::vector<std::vector<int>> weightMatrix;
public:
	Graph(std::vector<std::vector<int>> weightMatrix, int numVertex);
	~Graph();
	std::string to_string();
	CompactGraph Graph::toCompact();
	
};

