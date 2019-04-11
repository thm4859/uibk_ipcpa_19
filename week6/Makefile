
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -g -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: count_seq count_omp count_ocl

count_seq: $(COMMON_DEPENDENCIES) count_seq.c
	@$(CC) $(CC_FLAGS) count_seq.c -o count_seq

count_omp: $(COMMON_DEPENDENCIES) count_omp.c
	@$(CC) $(CC_FLAGS) count_omp.c -o count_omp -fopenmp

count_ocl: $(COMMON_DEPENDENCIES) count_ocl.c
	@$(CC) $(CC_FLAGS) faulty_count_ocl.c -o faulty_count_ocl -lOpenCL

.PHONEY: clean
clean:
	@rm count_seq count_omp count_ocl
	
run: all
	@echo "Sequential:"
	@./count_seq
	@echo
	@echo "OpenMP:"
	@./count_omp
	@echo
	@echo "OpenCL:"
	@./faulty_count_ocl
