#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include "Graph.h"
#include "CompactGraph.h"
#include "DataStructureOnGpu.cuh"
#include "utilities.cuh"



void mst(Graph g);

void mst(CompactGraph g);

void mst(DatastructuresOnGpu onGPU);

