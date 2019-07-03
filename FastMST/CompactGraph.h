#include <string>
#include "Graph.h"

class CompactGraph
{
private:

public:
	std::vector<int> vertices;
	std::vector<int> edgePtr;
	int edgesCount = 0;
	std::vector<int>edges;
	std::vector<int>weights;

	CompactGraph(std::vector<std::vector<int>> weightMatrix);
	CompactGraph(Graph &g);
	~CompactGraph();

	void print();
};

