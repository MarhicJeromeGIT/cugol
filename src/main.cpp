# include "game_of_life.h"
# include <iostream>
using namespace std;

const int W = 10000; // Width of the grid
const int H = 10000; // Height of the grid

int main() {
    // Initialize an empty grid using dynamic allocation
    int* grid = new int[H * W]();
    int* duplicated = new int[H * W](); // Additional grid for duplication

    // Initialize a glider pattern
    grid[1 * W + 2] = 1;
    grid[2 * W + 3] = 1;
    grid[3 * W + 1] = 1;
    grid[3 * W + 2] = 1;
    grid[3 * W + 3] = 1;

    // Run one iteration of the game
    for(int i = 0; i < 30; i++) {
        step(grid, duplicated);
        // Uncomment the following line to print the grid
        // print(grid);
    }

    //print(grid);

    // Free the dynamically allocated memory
    delete[] grid;
    delete[] duplicated;

    return 0;
}
