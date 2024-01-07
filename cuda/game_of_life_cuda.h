#ifndef GAME_OF_LIFE_CUDA_H
#define GAME_OF_LIFE_CUDA_H

// CUDA Runtime
#include <cuda_runtime.h>

const int W = 10;
const int H = 10;

/**
 * Function to print the grid to the console.
 * @param grid The grid to be printed.
 */
void print(const int* grid);

__device__ int count_neighbors(int H, int* grid, int i, int j);

__global__ void swapGrids(int H, int W, int *grid1, int* grid2);

__global__ void step(int H, int W, int* src, int* dst);

#endif // GAME_OF_LIFE_CUDA_H
