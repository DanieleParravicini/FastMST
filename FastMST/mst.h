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


long long int mst(Graph &g);

long long int mst(CompactGraph &g);

long long int mst(DatastructuresOnGpu* onGPU);

long long int verifyMst(Graph &g);

long long int verifyMst(CompactGraph &g);

long long int verifyMst(DatastructuresOnGpu* onGPU);

