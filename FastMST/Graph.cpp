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
	bool flag_w = false, flag_v = false;
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

				if ((v & mask_v) != v || (u & mask_v) != u ) {
					flag_v = true;
									}
				if ((w & mask_w) != w) {
					flag_w = true;
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

		if(flag_v)
			std::cout << "Pay attention there exists some vertex id that has exceeded the representation power consider change bits dedicated." << std::endl;
		
		if(flag_w)
			std::cout << "Pay attention there exists some weight that has exceeded the representation power consider change bits dedicated." << std::endl;

	}
	else {
		std::cout << "Error while opening file";
	}

}

void generateRandom(int nr_vertices, int nr_edges , Graph& g) {
	boost::minstd_rand gen;
	int max_weight = 1000;

	
	//int nr_edges = nr_vertices * 4;
	// https://valelab4.ucsf.edu/svn/3rdpartypublic/boost/libs/graph_parallel/doc/html/rmat_generator.html
	// the four floating point numbers that follows: a, b, c, and d represent the probability that a generated edge 
	// is placed of each of the 4 quadrants of the partitioned adjacency matrix
	float prob_first = 0.25;
	float prob_second = 0.25;
	float prob_third = 0.25;
	float prob_fourth = 0.25;

	g = Graph(RMATGen(gen, nr_vertices,nr_edges, prob_first, prob_second, prob_third, prob_fourth), RMATGen(), nr_vertices);
	//
	//g = Graph(ERGen(gen, nr_vertices, 0.25), ERGen(), nr_vertices);
	std::pair<Graph::edge_iterator, Graph::edge_iterator> iterators = boost::edges(g);

	Graph::edge_iterator next;
	//eliminate self loops

	
	boost::remove_edge_if([&](Edge e) {
		return e.m_source == e.m_target;
	}, g);

	iterators = boost::edges(g);
	//the generators by default produces edges with weights = 0.
	//add weights
	for (; iterators.first != iterators.second; ++iterators.first) {
		int w; 
		//since the generator sometimes produces parralle arc to avoid that they experience different weights
		//first look for weights and if found != 0 leave that cost.
		std::pair<Edge, bool> already_inserted = boost::edge(iterators.first->m_source, iterators.first->m_target, g);
		assert(already_inserted.second);
		
		w = boost::get(boost::edge_weight_t(), g, already_inserted.first);
		if (w == 0) {
			w = (rand() % max_weight) +1;
		}
		boost::put(boost::edge_weight_t(), g, *iterators.first, w);
	}

	
	//make sure that al the elements are connected
	for (int i = 0; i < nr_vertices - 2; i++) {

		int w = max_weight;
		std::pair<Edge, bool> already_inserted = boost::edge(i, i+1, g);

		if (!already_inserted.second) {
			boost::add_edge(i , i+1 , w , g);
			
		}
	}
}

void printForWebgraphviz(Graph &g) {
	//small graph can be plotted at http://www.webgraphviz.com/

	std::cout << "graph { " << std::endl;
	std::pair<Graph::edge_iterator, Graph::edge_iterator> iterators = boost::edges(g);
	//substitute
	for (; iterators.first != iterators.second; ++iterators.first) {
	
			std::cout << iterators.first->m_source << " -- " << iterators.first->m_target << "[ label=\"" << boost::get(boost::edge_weight_t(), g, *iterators.first) << "\"]"<< std::endl;
		
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

void writeGraphToFile(Graph& g, std::ostream& out)
{
	WeightMap weights = boost::get(boost::edge_weight, g);
	//print problem description
	out << "c DIMACS graph" << std::endl;
	//print number of verts and edges
	out << "p sp " << boost::num_vertices(g) << " " << boost::num_edges(g) << std::endl; 
	//output the edges
	Graph::edge_iterator  ei, e_end;
	for (boost::tie(ei, e_end) = edges(g); ei != e_end; ++ei) {
		out << "a " << source(*ei, g) + 1 << " " << target(*ei, g) + 1 << " " << boost::get(boost::edge_weight_t(), g, *ei) << std::endl;
	}
}
