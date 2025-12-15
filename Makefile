# Compiler settings
CXX := g++
NVCC := nvcc


# Ripped this makefile from various sources and modified for my use.
# This was complete hell to make, setup, and debug.

ifneq ($(wildcard /usr/local/cuda/include/cuda_runtime.h),)
    CUDA_ROOT := /usr/local/cuda
    CUDA_INC_DIR := $(CUDA_ROOT)/include
    CUDA_LIB_DIR := $(CUDA_ROOT)/lib64
else ifneq ($(wildcard /usr/include/cuda_runtime.h),)
    CUDA_ROOT := /usr
    CUDA_INC_DIR := /usr/include
    CUDA_LIB_DIR := /usr/lib/x86_64-linux-gnu
else ifneq ($(wildcard /opt/cuda/include/cuda_runtime.h),)
    CUDA_ROOT := /opt/cuda
    CUDA_INC_DIR := /opt/cuda/include
    CUDA_LIB_DIR := /opt/cuda/lib64
else
    NVCC_PATH := $(shell command -v nvcc 2> /dev/null)
    ifneq ($(NVCC_PATH),)
        CUDA_ROOT := $(shell dirname $(shell dirname $(shell readlink -f $(NVCC_PATH))))
        CUDA_INC_DIR := $(CUDA_ROOT)/include
        CUDA_LIB_DIR := $(CUDA_ROOT)/lib64
    else
        $(error CUDA headers (cuda_runtime.h) not found. Please install CUDA or edit 'CUDA_ROOT' in this Makefile manually)
    endif
endif

# Onnx Runtime settings
ORT_ROOT := /home/thomas/Documents/ART_YOLO/cuda_onnx_inferencing/onnxruntime-linux-x64-gpu-1.23.2

ifneq ($(wildcard $(ORT_ROOT)/include/onnxruntime_cxx_api.h),)
    ORT_INCLUDE_DIR := $(ORT_ROOT)/include
    ORT_LIB_DIR := $(ORT_ROOT)/lib
else ifneq ($(wildcard /usr/include/onnxruntime/onnxruntime_cxx_api.h),)
    ORT_INCLUDE_DIR := /usr/include/onnxruntime
    ORT_LIB_DIR := /usr/lib
else ifneq ($(wildcard /usr/local/include/onnxruntime/onnxruntime_cxx_api.h),)
    ORT_INCLUDE_DIR := /usr/local/include/onnxruntime
    ORT_LIB_DIR := /usr/local/lib
else
    $(info [ERROR] ONNX Runtime header 'onnxruntime_cxx_api.h' NOT found.)
    $(info Current configured path: $(ORT_ROOT))
    $(error Please download ONNX Runtime (GPU version) and update ORT_ROOT in the Makefile)
endif

#BUILD FLAGS & TARGETS

# C++ Flags
CXXFLAGS := -std=c++17 -O3 -Wall -fopenmp -march=native -ffast-math
CXXFLAGS += -I./include -I$(ORT_INCLUDE_DIR) -I$(CUDA_INC_DIR)
CXXFLAGS += $(shell pkg-config --cflags opencv4)

# CUDA Flags (for NVCC)
NVCCFLAGS := -std=c++17 -O3 -use_fast_math -arch=sm_86 -I./include -I$(ORT_INCLUDE_DIR)
# NVCC needs to know where to find OpenCV headers if included in .cu files (not used here but good practice)
# NVCCFLAGS += --compiler-options "$(shell pkg-config --cflags opencv4)"

# Linker Flags
LDFLAGS := -L$(ORT_LIB_DIR) -L$(CUDA_LIB_DIR) -fopenmp
LDFLAGS += $(shell pkg-config --libs opencv4)
LDFLAGS += -lonnxruntime -lcudart -Wl,-rpath,$(ORT_LIB_DIR)

# Source files
SRC_DIR := src
OBJ_DIR := obj

# Find CPP and CU files
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS := $(wildcard $(SRC_DIR)/*.cu)

# Define Object files
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))

OBJS := $(CPP_OBJS) $(CU_OBJS)

# Binary name
TARGET := onnx_gpu

# Build Rules
all: info $(TARGET)

info:
	@echo "========================================"
	@echo "Build Configuration:"
	@echo "  CUDA Found at: $(CUDA_ROOT)"
	@echo "  ORT Found at:  $(ORT_INCLUDE_DIR)"
	@echo "========================================"

$(TARGET): $(OBJS)
	@echo "Linking..."
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build complete! Run with: ./$(TARGET) model.onnx image.jpg"

# Rule for C++ files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	@echo "Compiling C++ $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for CUDA files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	@echo "Compiling CUDA $<..."
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean info