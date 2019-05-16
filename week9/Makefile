
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: hillissteele

hillissteele: $(COMMON_DEPENDENCIES) hillissteele.c
	@$(CC) $(CC_FLAGS) hillissteele.c -o hillissteele -lOpenCL

.PHONEY: clean
clean:
	@rm hillissteele
	
run: all
	@echo "Hillis and Steele:"
	@./hillissteele



