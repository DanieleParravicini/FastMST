#include "Graph.h"
#include "Utils.h"

void loadGraphFromFile(std::string path, Graph& g) {
	//this procedure expect to receive data coming from .gr files encoded according to http://users.diag.uniroma1.it/challenge9/format.shtml
	//basically:
	//- c line of comment
	//- p sp <num of vertices> <num of edges>
	//- a <vertex1> <vertex2> <width>

	std::ifstream aFile;
	aFile.open(path, std::ios::in);
	char c;
	int mask_v = createMask(0, VERTEX_SIZE);
	int mask_w = createMask(0, WEIGHT_SIZE);

	int v, u, w;
	if (aFile.is_open()) {
		while (!aFile.eof())
		{
			aFile >> c;
			switch (c)
			{
			case('c') :
				aFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				break;
			case('p') :
				aFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				break;
			case('a') :
				aFile >> v;
				aFile >> u;
				aFile >> w;



				boost::add_edge(u & mask_v, v & mask_v, w & mask_w, g);
				break;
			default:
				break;
			}
		}
		aFile.close();
	}
	else {
		std::cout << "Error while opening file";
	}

}

void printForWebgraphviz(Graph &g) {
	//small graph can be plotted at http://www.webgraphviz.com/
	WeightMap weights = boost::get(boost::edge_weight, g);
	std::cout << "graph { " << std::endl;
	graph_traits<Graph>::vertex_iterator vertex, vertex_end;
	for (boost::tie(vertex, vertex_end) = boost::vertices(g); vertex != vertex_end; ++vertex) {

		graph_traits<Graph>::adjacency_iterator adj, adj_end;

		for (boost::tie(adj, adj_end) = boost::adjacent_vertices(*vertex, g); adj != adj_end; ++adj) {
			if (*vertex > *adj)
				continue;
			std::pair<Edge, bool> res = boost::edge(*vertex, *adj, g);
			std::cout << *vertex << " -- " << *adj << "[ label=\"" << boost::get(weights, res.first) << "\"]" << std::endl;
		}
	}

	std::cout << std::endl << "}";
}

