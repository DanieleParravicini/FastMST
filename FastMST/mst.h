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


int mst(Graph &g);

int mst(CompactGraph &g);

int mst(DatastructuresOnGpu* onGPU);

int verifyMst(DatastructuresOnGpu* onGPU);

