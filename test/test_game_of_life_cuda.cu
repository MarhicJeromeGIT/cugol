#include <gtest/gtest.h>
#include "../cuda/game_of_life_cuda.cuh" // Include your CUDA kernel header

TEST(CUDAKernelTest, StepKernelExecutionTest) {
    const int W = 5;
    const int H = 5;

    int *grid1, *grid2;
    cudaMallocManaged(&grid1, H * W * sizeof(int));
    cudaMallocManaged(&grid2, H * W * sizeof(int));

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

    // print_grid(W, H, grid1);

    // // Launch kernel and wait for completion
    step<<<1, 1>>>(H, W, grid1, grid2);
    cudaDeviceSynchronize();

    // print_grid(W, H, grid2);

    // // Check the result
    // Expected grid values after the kernel execution
    int expectedGrid[H * W] = {
        0, 0, 0, 0, 0, 
        0, 0, 0, 0, 0, 
        0, 1, 0, 1, 0, 
        0, 0, 1, 1, 0, 
        0, 0, 0, 0, 0
    };

    // Check each cell
    for(int i = 0; i < H; ++i) {
        for(int j = 0; j < W; ++j) {
            EXPECT_EQ(grid2[i * W + j], expectedGrid[i * W + j]);
        }
    }

    // // Free device memory
    cudaFree(grid1);
    cudaFree(grid2);
}
