
struct DatastructuresOnGpu {
	int* vertices = 0;
	int* edgePtr = 0;
	int* weights = 0;
	int* edges = 0;
	int numEdges;
	int numVertices;

	int* X;
	int* F;
	int* currentVertexMapping;
};