# Compiler settings
CXX = g++
CXXFLAGS = -Wall -g

# NVCC settings
NVCC = nvcc
NVCCFLAGS = # -arch=sm_35

# Google Test settings
GTEST_LIB = -lgtest -lgtest_main
PTHREAD_LIB = -pthread

# Project directory structure
SRC_DIR = ./src
CUDA_DIR = ./cuda
TEST_DIR = ./test
BUILD_DIR = ./build

# Source and Object files for the main application
MAIN_SRC = $(wildcard $(SRC_DIR)/*.cpp)
MAIN_OBJ = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(MAIN_SRC))

# CUDA source and object files
CUDA_SRC = $(wildcard $(CUDA_DIR)/*.cu)
CUDA_OBJ = $(patsubst $(CUDA_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRC))

# Source and Object files for the test (both C++ and CUDA)
TEST_CPP_SRC = $(wildcard $(TEST_DIR)/*.cpp)
TEST_CU_SRC = $(wildcard $(TEST_DIR)/*.cu)
TEST_CPP_OBJ = $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/test_%.o, $(TEST_CPP_SRC))
TEST_CU_OBJ = $(patsubst $(TEST_DIR)/%.cu, $(BUILD_DIR)/test_%.o, $(TEST_CU_SRC))

# Executable names
MAIN_EXEC = $(BUILD_DIR)/game_of_life
CUDA_MAIN_EXEC = $(BUILD_DIR)/game_of_life_cuda
TEST_EXEC = $(BUILD_DIR)/test_game_of_life
CUDA_TEST_EXEC = $(BUILD_DIR)/cuda_test_game_of_life

# Default target
all: $(MAIN_EXEC)
	./$(MAIN_EXEC)

# Main application build
$(MAIN_EXEC): $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# CUDA Main application build
cuda: $(CUDA_MAIN_EXEC)
	./$(CUDA_MAIN_EXEC)

$(CUDA_MAIN_EXEC): $(BUILD_DIR)/main_cuda.o $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Build object files for main application
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Build CUDA object files
$(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Test build and run
test: $(TEST_EXEC)
	./$(TEST_EXEC)

$(TEST_EXEC): $(TEST_CPP_OBJ) $(filter-out $(BUILD_DIR)/main.o, $(MAIN_OBJ))
	$(CXX) $(CXXFLAGS) -o $@ $^ $(GTEST_LIB) $(PTHREAD_LIB)

# Build object files for test
$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# CUDA Test build and run
cuda-test: $(CUDA_TEST_EXEC)
	./$(CUDA_TEST_EXEC)

$(CUDA_TEST_EXEC): $(TEST_CU_OBJ) $(filter-out $(BUILD_DIR)/main_cuda.o, $(CUDA_OBJ))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(GTEST_LIB)

# Build object files for CUDA tests
$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Clean up
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all cuda test cuda-test clean
