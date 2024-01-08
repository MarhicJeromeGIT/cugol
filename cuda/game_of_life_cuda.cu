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

__device__ int count_neighbors(int W, int* grid, int j, int i) {
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
  __shared__ int shared_mem[32 * 32];

  // Calculate the global position of the top-left cell of this block
  int globalX = blockIdx.x * 30;
  int globalY = blockIdx.y * 30;

  // Calculate the thread's global position
  int threadGlobalX = globalX + threadIdx.x;
  int threadGlobalY = globalY + threadIdx.y;
  int global_idx = threadGlobalY * W + threadGlobalX;

  // Load data into shared memory
  if (threadGlobalX < W && threadGlobalY < H) {
      shared_mem[threadIdx.y * 32 + threadIdx.x] = src[threadGlobalY * W + threadGlobalX];
  }

  __syncthreads();

  if(threadGlobalX >= W-1 || threadGlobalY >= H-1) {
      return;
  }
  // printf("global idx is %d, (%d, %d)\n", global_idx, threadGlobalX, threadGlobalY);


  if (threadIdx.x > 0 && threadIdx.x < 31 && threadIdx.y > 0 && threadIdx.y < 31) {
    int neighborsCount = count_neighbors(32, shared_mem, threadIdx.x, threadIdx.y);

    // For debugging
    // dst[global_idx] = neighborsCount; // shared_mem[threadIdx.y * 32 + threadIdx.x];

    int localIdx = threadIdx.y * 32 + threadIdx.x;
    if(shared_mem[localIdx] == 1 == 1) {
      if (neighborsCount == 2 || neighborsCount == 3) {
        // A live cell with 2 or 3 live neighbors survives
        dst[global_idx] = 1;
      } else {
        // A live cell with fewer than 2 or more than 3 live neighbors dies
        dst[global_idx] = 0;
      }
    } else {
      if (neighborsCount == 3) {
        // A dead cell with exactly 3 live neighbors becomes alive
        dst[global_idx] = 1;
      } else {
        // A dead cell with any other number of live neighbors remains dead
        dst[global_idx] = 0;
      }
    }
  }
}
