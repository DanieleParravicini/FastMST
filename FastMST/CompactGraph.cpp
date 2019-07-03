#include "CompactGraph.h"

CompactGraph::CompactGraph(std::vector<std::vector<int>> weightMatrix)
	:vertices(), edgePtr(), edges(), weights()
{
	int edgeCount = 0;
	for (int src = 0; src < weightMatrix.size(); src++) {
		vertices.push_back(src);

		edgePtr.push_back(edgesCount);

		for (int dst = 0; dst < weightMatrix[src].size(); dst++) {
			if (weightMatrix[src][dst] > 0) {
				edges.push_back(dst);
				weights.push_back(weightMatrix[src][dst]);
				edgesCount++;
			}
		}

	}
}


CompactGraph::CompactGraph(Graph &g)
	:vertices(), edgePtr(), edges(), weights()
{
	int edge_cnt = 0;
	WeightMap weights = boost::get(boost::edge_weight, g);

	graph_traits<Graph>::vertex_iterator vertex, vertex_end;
	for (boost::tie(vertex, vertex_end) = boost::vertices(g); vertex != vertex_end; ++vertex) {




		graph_traits<Graph>::adjacency_iterator adj, adj_end;
		boost::tie(adj, adj_end) = boost::adjacent_vertices(*vertex, g);
		if (adj == adj_end)
			continue; //this filters out vertex with no edges

		this->vertices.push_back(*vertex);
		this->edgePtr.push_back(edge_cnt);

		for (; adj != adj_end; ++adj) {
			std::pair<Edge, bool> res = boost::edge(*vertex, *adj, g);
			std::cout << *vertex << " -- " << *adj << "[" << boost::get(weights, res.first) << std::endl;
			this->edges.push_back(*adj);
			this->weights.push_back(boost::get(weights, res.first));
			edge_cnt++;
		}
	}

}

CompactGraph::~CompactGraph()
{
}


void CompactGraph::print() {


	std::cout << "Vertices:" << std::endl;
	for each (int i in this->vertices)
	{
		std::cout << "\t" << std::to_string(i);
	}

	std::cout << std::endl << "EdgePtr:" << std::endl;
	for each (int i in this->edgePtr)
	{
		std::cout << "\t" << std::to_string(i);
	}

	std::cout << std::endl << "Edges:" << std::endl;
	for each (int i in this->edges)
	{
		std::cout << "\t" << std::to_string(i);
	}
	std::cout << "Weights:" << std::endl;
	for each (int i in this->weights)
	{
		std::cout << "\t" << std::to_string(i);
	}
	std::cout << std::endl;

}