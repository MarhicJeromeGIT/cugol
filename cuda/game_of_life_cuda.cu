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

__device__ int count_neighbors(int W, int* grid, int row, int col) {
    int count = 0;

    // Check the 8 neighbors
    if (grid[(row-1) * W + col-1] == 1) count++;
    if (grid[(row-1) * W + col] == 1) count++;
    if (grid[(row-1) * W + col+1] == 1) count++;

    if (grid[row * W + col-1] == 1) count++;
    if (grid[row * W + col+1] == 1) count++;

    if (grid[(row+1) * W + col-1] == 1) count++;
    if (grid[(row+1) * W + col] == 1) count++;
    if (grid[(row+1) * W + col+1] == 1) count++;

    return count;
}

// Advance the simulation by one step
__global__ void step_shared_memory(int W, int H, int* src, int* dst) {
  const int windowSize = 32;
  __shared__ int shared_mem[windowSize * windowSize];

  // Calculate the global position of the top-left cell of this block
  int globalX = blockIdx.x * (windowSize-2);
  int globalY = blockIdx.y * (windowSize-2);

  // Calculate the thread's global position
  int threadGlobalX = globalX + threadIdx.x;
  int threadGlobalY = globalY + threadIdx.y;
  int global_idx = threadGlobalY * W + threadGlobalX;
  int localIdx = threadIdx.y * windowSize + threadIdx.x;

  // Load data into shared memory
  if (threadGlobalX < W && threadGlobalY < H) {
      shared_mem[localIdx] = src[threadGlobalY * W + threadGlobalX];
  }

  __syncthreads();

  if(threadGlobalX >= W-1 || threadGlobalY >= H-1) {
      return;
  }
  // printf("global idx is %d, (%d, %d)\n", global_idx, threadGlobalX, threadGlobalY);


  if (threadIdx.x > 0 && threadIdx.x < windowSize-1 && threadIdx.y > 0 && threadIdx.y < windowSize-1) {
    int neighborsCount = count_neighbors(windowSize, shared_mem, threadIdx.y, threadIdx.x);

    // For debugging
    // dst[global_idx] = neighborsCount; // shared_mem[localIdx];

    if(shared_mem[localIdx] == 1) {
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

// Advance the simulation by one step
__global__ void step_grid_stride(int W, int H, int* src, int* dst) {
    int thread_id = threadIdx.x;
    int stride = blockDim.x;
    int blockCount = gridDim.x;
    int block_index = blockIdx.x;

    // printf("thread_id: %d, stride: %d, blockCount: %d, block_index: %d\n", thread_id, stride, blockCount, block_index);

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