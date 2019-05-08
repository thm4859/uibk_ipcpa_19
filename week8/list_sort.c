#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "people.h"

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
    printf("Unsorted list\n");
    printf("----------------------------\n");
    person_t* list;
    // generate a list of N persons
    generate_list(&list, seed, N);  
    // print info
    print_list(list, N);
    person_t* sorted_list = count_sort(list, N);
    printf("----------------------------\n");
    printf("Sorted list\n");
    printf("----------------------------\n");
    print_list(sorted_list, N);

    // free the allocated memory
    free(list);
    free(sorted_list);

    return EXIT_SUCCESS;
}
