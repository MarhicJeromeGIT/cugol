# Compiler settings
CXX = g++
CXXFLAGS = -Wall -g

# Google Test settings
GTEST_LIB = -lgtest -lgtest_main -pthread

# Project directory structure
SRC_DIR = ./src
TEST_DIR = ./test
BUILD_DIR = ./build

# Source and Object files for the main application
MAIN_SRC = $(wildcard $(SRC_DIR)/*.cpp)
MAIN_OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(MAIN_SRC))

# Source and Object files for the test
TEST_SRC = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ = $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/test_%.o, $(TEST_SRC))

# Executable names
MAIN_EXEC = $(BUILD_DIR)/game_of_life
TEST_EXEC = $(BUILD_DIR)/test_game_of_life

# Default target
all: $(MAIN_EXEC)

# Main application build
$(MAIN_EXEC): $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build object files for main application
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Test build and run
test: $(TEST_EXEC)
	./$(TEST_EXEC)

$(TEST_EXEC): $(TEST_OBJ) $(filter-out $(BUILD_DIR)/main.o, $(MAIN_OBJ))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(GTEST_LIB)

# Build object files for test
$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean up
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all test clean
