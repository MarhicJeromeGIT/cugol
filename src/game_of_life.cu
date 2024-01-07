// Simple implement of the game of life in C++
// nvcc game_of_life.cu -o cugol && ./cugol
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

const int W = 10000;
const int H = 10000;

/**
 * Function to print the grid to the console.
 * @param grid The grid to be printed.
 */
void print(const int* grid) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++)
            cout << grid[i * W + j] << " ";
        cout << endl;
    }
    cout << "----" << endl;
}

__device__ int count_neighbors(int H, int* grid, int i, int j) {
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

__global__ void swapGrids(int H, int W, int *grid1, int* grid2) {
    // copy grid2 to grid1 and reset grid2 to 0
    // let's do a grid-stride loop
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int idx=thread_idx; idx < H * W; idx+=stride) {
        grid1[idx] = grid2[idx];
        grid2[idx] = 0;
    }
}

// Advance the simulation by one step
__global__
void step(int H, int W, int* src, int* dst) {
    int thread_id = threadIdx.x;
    int stride = blockDim.x;
    int blockCount = gridDim.x;
    int block_index = blockIdx.x;

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

int main() {
    // Initialize an empty grid:
    int *grid1, *grid2;
    cudaMallocManaged(&grid1, H * W * sizeof(int));
    cudaMallocManaged(&grid2, H * W * sizeof(int));
    // For now let's say that the edge cells are always empty (dead)

    for(int i=0; i < H; i++) {
        for(int j=0; j < W; j++) {
            grid1[i * W + j] = 0;
            grid2[i * W + j] = 0;
        }
    }

    // Initialize a glider:
    grid1[1 * W + 2] = 1;
    grid1[2 * W + 3] = 1;
    grid1[3 * W + 1] = 1;
    grid1[3 * W + 2] = 1;
    grid1[3 * W + 3] = 1;

    // print(grid1);
    int smCount; // We can set it to the number of SMs
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
    cout << "SM count : " << smCount << endl;

    for(int i=0; i < 30; i++) {
        step<<<smCount*4, 256>>>(H, W, grid1, grid2);
        cudaDeviceSynchronize();

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(1);
        }

        swapGrids<<<smCount*4, 256>>>(H, W, grid1, grid2);
        cudaDeviceSynchronize();
        // print(grid1);
    }

    // Free the allocated memory
    cudaFree(grid1);
    cudaFree(grid2);

    return 0;
}