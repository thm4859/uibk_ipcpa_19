#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MAX_AGE 120
#define NAME_LEN 32

typedef char name_t[NAME_LEN];

typedef struct {
	int age;
	name_t name;
} person_t;

// Name Generation ------------------------------------------------------------

#define BUF_SIZE 16
#define FIRST_NAMES_FILE "first_names.txt"
#define LAST_NAMES_FILE "last_names.txt"

int count_lines(const char *filename) {
	int lines = 0;
	FILE *f = fopen(filename, "r");
	assert(f != NULL);
	char c;
	while((c = fgetc(f)) != EOF) if(c == '\n') lines++;
	fclose(f);
	if(c != '\n') lines++;
	return lines;
}

int load_names(const char *filename, char ***storage) {
	int lines = count_lines(filename) - 1;
	*storage = (char**)malloc(lines * sizeof(char*));
	char *space =  (char*)malloc(lines * BUF_SIZE * sizeof(char));
	
	FILE *f = fopen(filename, "r");
	for(int i=0; i<lines; ++i) {
		(*storage)[i] = space + i * BUF_SIZE;
		assert(fgets((*storage)[i], BUF_SIZE-1, f) != NULL);
		// remove whitespace chars, if any
		char *c;
		while((c = strchr((*storage)[i], '\n'))) *c = '\0';
		while((c = strchr((*storage)[i], '\r'))) *c = '\0';
		while((c = strchr((*storage)[i], ' '))) *c = '\0';
	}
	fclose(f);
	return lines;
}

void gen_name(name_t buffer) {
	static char** first_names = NULL;
	static char** last_names = NULL;
	static int first_name_count, last_name_count;
	if(first_names == NULL) { // initialize on first call
		first_name_count = load_names(FIRST_NAMES_FILE, &first_names);
		last_name_count = load_names(LAST_NAMES_FILE, &last_names);
	}

	snprintf(buffer, NAME_LEN, "%s %s",
		first_names[rand()%first_name_count], 
		last_names[rand()%last_name_count]);
}

int random(int min, int max){
   return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

void generate_list(person_t** list, long seed, int entries) {
    // allocate a list of N persons
    *list = (person_t*)malloc(sizeof(person_t)*entries);
    // seed the random generator
    srand((unsigned) seed);
    for (int i = 0; i < entries; i++) {
        (*list)[i].age = random(0, MAX_AGE);
        gen_name((*list)[i].name);
    }
}

void print_list(person_t* list, int entries) {
    for (int i = 0; i < entries; i++) {
        printf("%d | %s\n", list[i].age, list[i].name);
    }
}

person_t *copy_list(person_t* list, int n) {
    person_t* A = (person_t*)malloc(sizeof(person_t)*n);
    for (int i = 0; i < n; i++) {
        strcpy(A[i].name, list[i].name);
        A[i].age = list[i].age;
    }
    return A;
}

person_t* count_sort(person_t* list, int n) {
    // start algorithm
    int k = MAX_AGE+1;
    person_t* A = list;
    int* C = malloc(sizeof(int)*k);
    // init C
    for (int i = 0; i < k; i++) {
        C[i] = 0;
    }
    // create histogram
    for (int i = 0; i < n; i++) {
        C[A[i].age] += 1;
    }
    // calculate address (prefix sum shifted)
    for (int i = 1; i < k; i++) {
        C[i] += C[i-1];
    }
    // B
    person_t* B = (person_t*)malloc(sizeof(person_t)*n);
    // copy each a value to b with position from c
    for (int i = 0; i < n; i++) {
        C[A[i].age] -= 1;
        B[C[A[i].age]] = A[i];
    }
    // free the allocated memory
    free(C);
    return B;
}
