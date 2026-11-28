# Compilers
NVCC = /usr/local/cuda/bin/nvcc
CC = gcc

# Directories
UTILITIES = ../../utilities

# Source and executable
EXERCISE ?= correlation_uvm.cu
TARGET = $(EXERCISE:.cu=.exe)

# Dataset configuration
DATASET ?= STANDARD_DATASET

ifeq ($(DATASET),STANDARD_DATASET)
    DATASET_FLAG =
else
    DATASET_FLAG = -D$(DATASET)
endif

# Compilation flags
CFLAGS = -O3 -I$(UTILITIES) -DPOLYBENCH_TIME $(DATASET_FLAG)
NVCCFLAGS = -O3 -arch=sm_53 -I$(UTILITIES) -DPOLYBENCH_TIME $(DATASET_FLAG)

# Default target
all: $(TARGET)

# Compile PolyBench
polybench.o: $(UTILITIES)/polybench.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA
$(TARGET): $(EXERCISE) correlation.h polybench.o
	$(NVCC) $(NVCCFLAGS) $(EXERCISE) polybench.o -o $@ -lm

# Execute
run: all
	./$(TARGET)

# Profile
profile: all
	sudo /usr/local/cuda/bin/nvprof --unified-memory-profiling off ./$(TARGET)

# Clean
clean:
	rm -f *.o *.exe

.PHONY: all run profile clean