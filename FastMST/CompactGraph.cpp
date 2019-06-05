#include <vector>
#include "CompactGraph.h"
#include <string>

CompactGraph::CompactGraph(std::vector<std::vector<int>> weightMatrix)
	:vertices(), edgePtr(), edges(), weights()
{
	int edgeCount = 0;
	for (int src = 0; src < weightMatrix.size(); src++) {
		vertices.push_back(src);
		if (src == 0) {
			//first vertex points directly to the start of the edge/weight vector
			edgePtr.push_back(0);
		}
		else {
			edgePtr.push_back(edgesCount);
		}
		for (int dst = 0; dst < weightMatrix[src].size(); dst++) {
			if (weightMatrix[src][dst] > 0) {
				edges.push_back(dst);
				weights.push_back(weightMatrix[src][dst]);
				edgesCount++;
			}
		}

	}
}


CompactGraph::~CompactGraph()
{
}


std::string CompactGraph::to_string() {
	std::string str;

	str.append("Vertices:\n");
	for each (int i in this->vertices)
	{
		str.append("\t" + std::to_string(i));
	}
	str.append("\n");
	str.append("EdgePtr:\n");
	for each (int i in this->edgePtr)
	{
		str.append("\t" + std::to_string(i));
	}
	str.append("\n");
	str.append("Edges:\n");
	for each (int i in this->edges)
	{
		str.append("\t" + std::to_string(i));
	}
	str.append("\n");
	str.append("Weights:\n");
	for each (int i in this->weights)
	{
		str.append("\t" + std::to_string(i));
	}
	str.append("\n");
	return str;
}
