# include "../cuda/game_of_life_cuda.cuh"
# include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
    int W = 32; // Width of the grid
    int H = 32; // Height of the grid
    int iterations = 20; // Default iteration count
    bool useSharedMemory = true; // otherwise it's the stride loop implementation

    // Parse optional command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" && i + 1 < argc) {
            W = std::atoi(argv[++i]);
        } else if (arg == "-h" && i + 1 < argc) {
            H = std::atoi(argv[++i]);
        } else if (arg == "-iteration" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (arg == "-shared" && i + 1 < argc) {
            useSharedMemory = std::atoi(argv[++i]);
        }
    }

    std::cout << "Width: " << W << ", Height: " << H << ", Iterations: " << iterations << std::endl;

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

    dim3 blocks, threadsPerBlock;
    if(useSharedMemory) {
        const int tc = 16; // thread count
        // So we launch blocks of 16*16 threads
        // but the threads corresponding to the edge cells will do nothing.
        cout << "Using shared memory" << endl;
        blocks = dim3((W + tc-1) / (tc-2), (H + tc-1) / (tc-2)); // Adjust to cover the entire grid
        threadsPerBlock = dim3(tc, tc);
    } else {
        cout << "Using stride loop" << endl;
        blocks = dim3(smCount*4);
        threadsPerBlock = dim3(256);

        // for debug
        // blocks = dim3(1);
        // threadsPerBlock = dim3(1);
    }
    printf("Launching Blocks: %d, %d\n", blocks.x, blocks.y);
    printf("Launching Threads per Block: %d, %d\n", threadsPerBlock.x, threadsPerBlock.y);

    for(int i=0; i < iterations; i++) {
        if (useSharedMemory) {
            // Launch the shared memory version of the kernel
            step_shared_memory<<<blocks, threadsPerBlock>>>(W, H, grid1, grid2);
        } else {
            // Launch the grid-stride version of the kernel
            step_grid_stride<<<blocks, threadsPerBlock>>>(W, H, grid1, grid2);
        }
        cudaDeviceSynchronize();

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(1);
        }

        swapGrids<<<smCount*4, 256>>>(W, H, grid1, grid2);
        cudaDeviceSynchronize();
        // print_grid(W, H, grid1);
    }

    // print_grid(W, H, grid1);

    // Free the allocated memory
    cudaFree(grid1);
    cudaFree(grid2);

    return 0;
}