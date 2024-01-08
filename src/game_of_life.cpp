#include <iostream>
#include "game_of_life.h"
using namespace std;

void print(const int8_t* grid, int W, int H) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            cout << grid[i * W + j] << " ";
        }
        cout << endl;
    }
    cout << "----" << endl;
}

// Note: we assume it's never called on the edges/corners of the grid
// otherwise this will go off bounds
int count_neighbors(const int8_t* grid, int i, int j, int W, int H) {
    if(i==0 || j==0 || i==H-1 || j==W-1) {
        cout << "ERROR: count_neighbors called on the edges of the grid (i=" << i << ", j=" << j << ")" << endl;
        return -1;
    }

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

void step(int8_t* grid, int8_t* duplicated, int W, int H) {
    // Copy the current grid state to the duplicated grid
    for (int i = 0; i < H * W; i++) {
        duplicated[i] = grid[i];
    }

    // Apply the rules of the Game of Life
    for(int i = 1; i < H - 1; i++) {
        for(int j = 1; j < W - 1; j++) {
            int neighborsCount = count_neighbors(duplicated, i, j, W, H);

            // A dead cell becomes alive if it has exactly three live neighbors
            if (duplicated[i * W + j] == 0 && neighborsCount == 3)
                grid[i * W + j] = 1;
            // A live cell with fewer than two or more than three live neighbors dies
            else if (duplicated[i * W + j] == 1 && (neighborsCount < 2 || neighborsCount > 3))
                grid[i * W + j] = 0;
        }
    }
}
