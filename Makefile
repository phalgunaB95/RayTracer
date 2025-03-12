# Compiler
NVCC = nvcc

# Default build mode (can be overridden via command line)
BUILD_MODE ?= RELEASE

# Compiler flags for each mode
CFLAGS_DEBUG = -std=c++17 -g -G -DDEBUG
CFLAGS_RELEASE = -std=c++17 -O3

# Choose flags based on the build mode
ifeq ($(BUILD_MODE), DEBUG)
    CFLAGS = $(CFLAGS_DEBUG)
else
    CFLAGS = $(CFLAGS_RELEASE)
endif

# Directories
SRC_DIR = src
BUILD_DIR = build
TARGET = raytracer

# Source and object files
SOURCES = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cc)
OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(SOURCES:.cc=.o))

# Build target
$(TARGET): $(OBJECTS)
	@echo "Linking: $@ (Mode: $(BUILD_MODE))"
	$(NVCC) $(CFLAGS) -o $@ $^

# Compile .cu files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling CUDA source: $< (Mode: $(BUILD_MODE))"
	$(NVCC) $(CFLAGS) -c $< -o $@

# Compile .cc files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(BUILD_DIR)
	@echo "Compiling C++ source: $< (Mode: $(BUILD_MODE))"
	$(NVCC) $(CFLAGS) -x cu -c $< -o $@

# Clean up build and output files
clean:
	@echo "Cleaning up..."
	rm -rf $(BUILD_DIR) $(TARGET)

all: $(TARGET)

# Phony targets
.PHONY: clean
