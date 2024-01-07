#include <iostream>
#include "game_of_life.h"
using namespace std;

void print(const int* grid, int W, int H) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            cout << grid[i * W + j] << " ";
        }
        cout << endl;
    }
    cout << "----" << endl;
}

int count_neighbors(const int* grid, int i, int j, int W, int H) {
    int count = 0;

    // Iterate through the neighbors
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            // Skip the cell itself
            if (x == 0 && y == 0) continue;

            // Calculate neighbor's position
            int ni = i + x;
            int nj = j + y;

            // Check if the neighbor is within the bounds of the grid
            if (ni >= 0 && ni < H && nj >= 0 && nj < W) {
                // Count if the neighbor is alive
                if (grid[ni * W + nj] == 1) count++;
            }
        }
    }

    return count;
}

void step(int* grid, int* duplicated, int W, int H) {
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
