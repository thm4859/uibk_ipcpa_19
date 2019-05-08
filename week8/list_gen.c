#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "people.h"

void generate_list(person_t** list, long seed, int entries);
void print_list(person_t* list, int entries);
int random(int min, int max);

int main(int argc, char** argv) {
    
    // read N from input
    int N = 10;
    long seed = 42;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        seed = atol(argv[2]);
    }
    printf("N = %d\n", N);
    printf("seed = %ld\n", seed);
    printf("----------------------------\n");
    
    person_t* list;
    // generate a list of N persons
    generate_list(&list, seed, N);  
    // print info
    print_list(list, N);
    // free the allocated memory
    free(list);

    return EXIT_SUCCESS;
}

void generate_list(person_t** list, long seed, int entries) {
    // allocate a list of N persons
    *list = (person_t*)malloc(sizeof(person_t)*entries);
    // seed the random generator
    srand((unsigned) seed);
    for (int i = 0; i < entries; i++) {
        (*list)[i].age = random(1, MAX_AGE);
        gen_name((*list)[i].name);
    }
}

void print_list(person_t* list, int entries) {
    for (int i = 0; i < entries; i++) {
        printf("%d | %s\n", list[i].age, list[i].name);
    }
}

int random(int min, int max){
   return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}
