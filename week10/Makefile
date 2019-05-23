
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: hillissteele downsweep prefixglobal

hillissteele: $(COMMON_DEPENDENCIES) hillissteele.c
	@$(CC) $(CC_FLAGS) hillissteele.c -o hillissteele -lOpenCL
	
downsweep: $(COMMON_DEPENDENCIES) downsweep.c
	@$(CC) $(CC_FLAGS) downsweep.c -o downsweep -lOpenCL

prefixglobal: $(COMMON_DEPENDENCIES) prefixglobal.c
	@$(CC) $(CC_FLAGS) prefixglobal.c -o prefixglobal -lOpenCL

.PHONEY: clean
clean:
	@rm hillissteele
	@rm downsweep
	@rm prefixglobal
	
run: all
	@echo "Hillis and Steele:"
	@./hillissteele
	@echo ""
	@echo "Downsweep:"
	@./downsweep	
	@echo ""
	@echo "Prefix global:"
	@./prefixglobal


