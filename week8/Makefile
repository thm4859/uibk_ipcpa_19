
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -g -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: list_gen

list_gen: $(COMMON_DEPENDENCIES) list_gen.c
	@$(CC) $(CC_FLAGS) list_gen.c -o list_gen


.PHONEY: clean
clean:
	@rm list_gen
	
run: all
	@echo "list_gen:"
	@./list_gen
