#ifndef GAME_OF_LIFE_CUDA_H
#define GAME_OF_LIFE_CUDA_H

// CUDA Runtime
#include <cuda_runtime.h>

/**
 * Function to print the grid to the console.
 * @param grid The grid to be printed.
 */
void print_grid(int W, int H, const int* grid);

__device__ int count_neighbors(int W, int* grid, int i, int j);

__global__ void swapGrids(int W, int H, int *grid1, int* grid2);

__global__ void step(int W, int H, int* src, int* dst);

#endif // GAME_OF_LIFE_CUDA_H
