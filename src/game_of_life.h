#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

/**
 * Function to count the number of alive neighbors of a cell.
 * @param grid The grid.
 * @param i The row index of the cell.
 * @param j The column index of the cell.
 * @param W The width of the grid.
 * @return The count of alive neighbors.
 */
int count_neighbors(const int* grid, int i, int j, int W, int H);

#endif // GAME_OF_LIFE_H
