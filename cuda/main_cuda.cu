# include "../cuda/game_of_life_cuda.cuh"
# include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
    int W = 10; // Width of the grid
    int H = 10; // Height of the grid
    int iterations = 30; // Default iteration count

    // Parse optional command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" && i + 1 < argc) {
            W = std::atoi(argv[++i]);
        } else if (arg == "-h" && i + 1 < argc) {
            H = std::atoi(argv[++i]);
        } else if (arg == "-iteration" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
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

    for(int i=0; i < iterations; i++) {
        step<<<smCount*4, 256>>>(H, W, grid1, grid2);
        cudaDeviceSynchronize();

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(1);
        }

        swapGrids<<<smCount*4, 256>>>(H, W, grid1, grid2);
        cudaDeviceSynchronize();
        // print_grid(W, H, grid1);
    }

    // print_grid(W, H, grid1);

    // Free the allocated memory
    cudaFree(grid1);
    cudaFree(grid2);

    return 0;
}