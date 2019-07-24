#pragma once
#include "Graph.h"
#include "CompactGraph.h"
#include "DataStructureOnGpu.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>

#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include <thrust\device_vector.h>
#include <thrust\scan.h>
#include <thrust\sort.h>

#include "Utils.h"
#include "mstKernels.cuh"


long long int mst(Graph &g, std::vector<Edge> &mst_resul);

long long int mst(CompactGraph &g, unsigned int * array_mst_result, unsigned int * lenght_array_mst_result);

long long int mst(DatastructuresOnGpu* onGPU);

long long int verifyMst(Graph &g, std::vector<Edge> &mst_resul);

long long int verifyMst(CompactGraph &g, unsigned int * array_mst_result, unsigned int * lenght_array_mst_result);

long long int verifyMst(DatastructuresOnGpu* onGPU);

