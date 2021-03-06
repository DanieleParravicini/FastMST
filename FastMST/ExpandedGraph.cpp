#include "ExpandedGraph.h"

//TODO: assumed 1 connected component? in other words that node has always a outgoing edge?
ExpandedGraph::ExpandedGraph(std::vector<std::vector<int>> weightMatrix)
	:vertices(), edgePtr(), edges(), weights(), edgesIds()
{
	int edgeCount = 0;
	for (int src = 0; src < weightMatrix.size(); src++) {
		

		edgePtr.push_back(edgesCount);

		for (int dst = 0; dst < weightMatrix[src].size(); dst++) {
			if (weightMatrix[src][dst] > 0) {
				vertices.push_back(src);
				edges.push_back(dst);
				weights.push_back(weightMatrix[src][dst]);
				edgesCount++;
			}
		}

	}
}


ExpandedGraph::ExpandedGraph(Graph &g)
	:vertices(), edgePtr(), edges(), weights(), edgesIds()
{
	
	std::vector<unsigned int> map;
	WeightMap weights = boost::get(boost::edge_weight, g);
	//1. build a map by inserting node with at least an exiting arc.
	graph_traits<Graph>::vertex_iterator vertex, vertex_end;


	unsigned int v = 0;
	int edge_cnt = 0;

	std::vector<unsigned int>::iterator i;
	for (boost::tie(vertex, vertex_end) = boost::vertices(g); vertex != vertex_end; ++vertex) {
		graph_traits<Graph>::adjacency_iterator adj, adj_end;
		boost::tie(adj, adj_end) = boost::adjacent_vertices(*vertex, g);
		if (adj == adj_end)
			continue; //this filters out vertex with no edges

		
		
		this->edgePtr.push_back(edge_cnt);

		for (; adj != adj_end; ++adj) {
			std::pair<Edge, bool> res = boost::edge(*vertex, *adj, g);
			std::pair<Edge, bool> res2 = boost::edge(*adj, *vertex, g);
			assert(!res.second || res2.second);

			this->vertices.push_back(*vertex);
			this->edges.push_back(*adj);
			int w = boost::get(weights, res.first);
			this->weights.push_back(w);
			this->edgesIds.push_back(edge_cnt);
			edge_cnt++;
			
		}
		v++;
	}

}

ExpandedGraph::~ExpandedGraph()
{
}


void ExpandedGraph::print() {


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
