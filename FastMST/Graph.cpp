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
				//p sp n m where n is number of nodes, m the number of arcs.
				aFile.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
				aFile.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
				aFile >> v;
				g = Graph(v);
				aFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				
				break;
			case('a') : {
				aFile >> v;
				aFile >> u;
				aFile >> w;
				v--;
				u--;

				if ((v & mask_v) != v || (u & mask_v) != u || (w & mask_w) != w) {
					std::cout << "Pay attention there exists some vector or weight that has exceeded the representation power consider change bits dedicated." << std::endl;
				}
				//the Undirected graph already conside u--w-->v and v--w-->u at the same way returning also the same weights.
				//no need to insert the edge if it is already thereS
				std::pair<Edge, bool> already_inserted = boost::edge(v, u, g);
				if (!already_inserted.second) {
					boost::add_edge(v & mask_v, u & mask_v, w & mask_w, g);
					//boost::add_edge(u & mask_v, v & mask_v, w & mask_w, g);
				}
				
				aFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				break;
			}
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
			
			std::pair<Edge, bool> res = boost::edge(*vertex, *adj, g);
			std::cout << *vertex << " -- " << *adj << "[ label=\"" << boost::get(weights, res.first) << "\"];";// << std::endl;
		}
	}

	std::cout << std::endl << "}" << std::endl;
}

void toGraph(Graph &g, DatastructuresOnGpu* onGPU) {
	unsigned int
		*e = (unsigned int*)malloc(sizeof(unsigned int)* onGPU->numEdges),
		*w = (unsigned int*)malloc(sizeof(unsigned int)* onGPU->numEdges),
		*e_ptr = (unsigned int*)malloc(sizeof(unsigned int)* onGPU->numVertices);

	cudaMemcpy(e,		onGPU->edges, sizeof(unsigned int) * onGPU->numEdges, cudaMemcpyDeviceToHost);
	cudaMemcpy(w,		onGPU->weights, sizeof(unsigned int) * onGPU->numEdges, cudaMemcpyDeviceToHost);
	cudaMemcpy(e_ptr,	onGPU->edgePtr, sizeof(unsigned int) * onGPU->numVertices, cudaMemcpyDeviceToHost);


	for (unsigned int i = 0; i < onGPU->numVertices - 1; i++) {
		for (unsigned int v = e_ptr[i]; v < e_ptr[i + 1]; v++) {
			boost::add_edge(i, e[v], w[v], g);
		}
	}
	for (unsigned int v = e_ptr[onGPU->numVertices - 1]; v < onGPU->numEdges; v++) {
		boost::add_edge(onGPU->numVertices - 1, e[v], w[v], g);

	}


	free(e);
	free(w);
	free(e_ptr);
}
