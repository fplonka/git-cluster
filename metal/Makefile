# Makefile

# Compiler settings
CXX = clang++
CXXFLAGS = -std=c++17 -framework Metal -framework Foundation -framework MetalKit -I/Users/filip/Downloads/metal-cpp -I/usr/local/include
# LDFLAGS = -L/usr/local/lib -lcnpy
LDFLAGS = -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lcnpy

METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

# Paths
SRC_DIR = src
BUILD_DIR = build

# Files
METAL_SRC = $(SRC_DIR)/add.metal
METAL_AIR = $(BUILD_DIR)/add.air
METAL_LIB = $(BUILD_DIR)/default.metallib
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(CPP_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
EXECUTABLE = $(BUILD_DIR)/MetalAdderApp

# Metal shader compilation
$(METAL_LIB): $(METAL_AIR)
	$(METALLIB) -o $@ $<

$(METAL_AIR): $(METAL_SRC)
	mkdir -p $(BUILD_DIR)
	$(METAL) -c $< -o $@

# C++ compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Executable
$(EXECUTABLE): $(OBJECTS) $(METAL_LIB)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS)

# Phony targets
.PHONY: all clean

all: $(EXECUTABLE)

clean:
	rm -rf $(BUILD_DIR)
