// Simple implement of the game of life in C++
// nvcc game_of_life.cu -o cugol && ./cugol
#include <iostream>
#include "game_of_life_cuda.cuh"
using namespace std;

// TODO: It's not a CUDA method, just a helper, could be in src/ maybe
void print_grid(int W, int H, const int* grid) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++)
            cout << grid[i * W + j] << " ";
        cout << endl;
    }
    cout << "----" << endl;
}

__global__ void swapGrids(int W, int H, int *grid1, int* grid2) {
    // copy grid2 to grid1 and reset grid2 to 0
    // let's do a grid-stride loop
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int idx=thread_idx; idx < H * W; idx+=stride) {
        grid1[idx] = grid2[idx];
        grid2[idx] = 0;
    }
}

__device__ int count_neighbors(int W, int* grid, int i, int j) {
    int count = 0;
    // Check the 8 neighbors
    if (grid[(i-1) * W + j-1] == 1) count++;
    if (grid[(i-1) * W + j] == 1) count++;
    if (grid[(i-1) * W + j+1] == 1) count++;

    if (grid[i * W + j-1] == 1) count++;
    if (grid[i * W + j+1] == 1) count++;

    if (grid[(i+1) * W + j-1] == 1) count++;
    if (grid[(i+1) * W + j] == 1) count++;
    if (grid[(i+1) * W + j+1] == 1) count++;

    return count;
}


// Advance the simulation by one step
__global__ void step(int W, int H, int* src, int* dst) {
    int thread_id = threadIdx.x;
    int stride = blockDim.x;
    int blockCount = gridDim.x;
    int block_index = blockIdx.x;

    printf("thread_id: %d, stride: %d, blockCount: %d, block_index: %d\n", thread_id, stride, blockCount, block_index);

    // Rules
    // A dead cell becomes alive if it has exactly three live neighbors. This simulates reproduction.
    // A live cell with two or three live neighbors continues to live. This simulates a balanced environment.
    // A live cell with fewer than two live neighbors dies (underpopulation) due to isolation.
    // A live cell with more than three live neighbors dies (overpopulation) due to limited resources.
    for(int i= 1 + block_index; i < H-1; i += blockCount) { // for each row
        for(int j= 1 + thread_id; j < W-1; j+= stride) { // for each column, but we have stride equal to the block dim
            int idx = i * W + j;

            int neighborsCount = count_neighbors(H, src, i, j);
            if (src[idx] == 1) {
                if (neighborsCount == 2 || neighborsCount == 3) {
                    // A live cell with 2 or 3 live neighbors survives
                    dst[idx] = 1;
                } else {
                    // A live cell with fewer than 2 or more than 3 live neighbors dies
                    dst[idx] = 0;
                }
            } else {
                if (neighborsCount == 3) {
                    // A dead cell with exactly 3 live neighbors becomes alive
                    dst[idx] = 1;
                } else {
                    // A dead cell with any other number of live neighbors remains dead
                    dst[idx] = 0;
                }
            }
        }
    }
}
