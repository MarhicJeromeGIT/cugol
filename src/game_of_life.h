#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include <cstdint>

/**
 * Function to count the number of alive neighbors of a cell.
 * @param grid The grid.
 * @param i The row index of the cell.
 * @param j The column index of the cell.
 * @param W The width of the grid.
 * @return The count of alive neighbors.
 */
int count_neighbors(const int8_t* grid, int i, int j, int W, int H);

/**
 * Function to print the grid to the console.
 * @param grid The grid to be printed.
 */
void print(const int8_t* grid, int W, int H);

/**
 * Function to update the grid to the next state based on Game of Life rules.
 * @param grid The current state of the grid.
 * @param duplicated A duplicate of the grid used for reading current states.
 */
void step(int8_t* grid, int8_t* duplicated, int W, int H);

#endif // GAME_OF_LIFE_H
