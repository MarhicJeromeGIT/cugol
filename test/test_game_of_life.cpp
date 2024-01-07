#include "../src/game_of_life.h"
#include <gtest/gtest.h>

class NeighborsTest : public ::testing::Test {
protected:
    int* grid;

    virtual void SetUp() {
        const int W = 3;
        const int H = 3;
        // Initialize a 3x3 grid
        grid = new int[W * H]{0};

        // Example setup:
        // 0 1 0
        // 1 0 1
        // 0 1 0
        grid[1] = 1;
        grid[3] = 1;
        grid[5] = 1;
        grid[7] = 1;
    }

    virtual void TearDown() {
        delete[] grid;
    }
};

TEST_F(NeighborsTest, CenterCell) {
    // Testing the center cell (1,1), expecting 4 neighbors
    EXPECT_EQ(4, count_neighbors(grid, 1, 1, 3, 3));
}

TEST_F(NeighborsTest, CornerCell) {
    // Testing a corner cell (0,0), expecting 2 neighbors
    EXPECT_EQ(2, count_neighbors(grid, 0, 0, 3, 3));
}

TEST_F(NeighborsTest, EdgeCell) {
    // Testing an edge cell (0,1), expecting 2 neighbors
    EXPECT_EQ(2, count_neighbors(grid, 0, 1, 3, 3));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}