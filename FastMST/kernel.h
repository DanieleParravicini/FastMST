#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include "Graph.h"
#include "CompactGraph.h"
#include "DataStructureOnGpu.cuh"
#include "utilities.cuh"

#include <math.h>

#include <thrust\device_vector.h>
#include <thrust\scan.h>
#include <thrust\sort.h>



void mst(Graph g);

void mst(CompactGraph g);

void mst(DatastructuresOnGpu onGPU);

void minOutgoingEdge(DatastructuresOnGpu onGPU);
