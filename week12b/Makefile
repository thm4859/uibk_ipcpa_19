
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -march=native -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: chain_matmul_ocl

chain_matmul_ocl: $(COMMON_DEPENDENCIES) chain_matmul.c cl_utils.h
	@$(CC) $(CC_FLAGS) chain_matmul.c -o chain_matmul_ocl -lOpenCL -lm -fopenmp

.PHONEY: clean
clean:
	@rm chain_matmul_ocl
	
run: all
	@echo "Running........"
	@./chain_matmul_ocl
