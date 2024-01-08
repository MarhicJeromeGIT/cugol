# include "game_of_life.h"
# include <iostream>
# include <cstdint>
using namespace std;

int main(int argc, char* argv[]) {
    int W = 10000; // Width of the grid
    int H = 10000; // Height of the grid
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

    // Initialize an empty grid using dynamic allocation
    int8_t* grid = new int8_t[H * W]();
    int8_t* duplicated = new int8_t[H * W](); // Additional grid for duplication

    // Initialize a glider pattern
    grid[1 * W + 2] = 1;
    grid[2 * W + 3] = 1;
    grid[3 * W + 1] = 1;
    grid[3 * W + 2] = 1;
    grid[3 * W + 3] = 1;

    for(int i = 0; i < iterations; i++) {
        step(grid, duplicated, W, H);
        // Uncomment the following line to print the grid
        // print(grid);
    }

    // print(grid, W, H);

    // Free the dynamically allocated memory
    delete[] grid;
    delete[] duplicated;

    return 0;
}
