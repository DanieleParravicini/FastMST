#include "Graph.h"
#include <vector>
#include <string>
#include "CompactGraph.h"

Graph::Graph(std::vector<std::vector<int>> weights, int numVertex)
	: weightMatrix(weights)
{
	if (weightMatrix.size() != numVertex) {
		throw std::string("weights dimensions must be (" +
			std::to_string(numVertex) + "," + std::to_string(numVertex) + ")"
			);
	}
	for each (std::vector<int> v in weightMatrix)
	{
		if (weightMatrix.size() != numVertex) {
			throw std::string("weights dimensions must be (" +
				std::to_string(numVertex) + "," + std::to_string(numVertex) + ")"
				);
		}
	}
}



Graph::~Graph() {

}

std::string Graph::to_string( )
{
	std::string str;
	str.append("{\n");
	for each (std::vector<int> vector in weightMatrix)
	{
		for each(int value in vector) {
			str.append("\t" + std::to_string(value));
		}
		str.append("\n");
	}
	str.append("}\n");
	return str;
}

CompactGraph Graph::toCompact()
{
	std::vector<std::vector<int>> copyWeights(weightMatrix);
	return CompactGraph(copyWeights);
}
