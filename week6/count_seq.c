#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    
    // read N from input
    long long N = 100*1000*1000;
    if (argc > 1) {
        N = atoll(argv[1]);
    }
    printf("N = %lld\n", N);

    // create an array of N bytes
    char* array = malloc(sizeof(char)*N);
   
    // seed the random generator
    srand((unsigned) time(NULL));

    // fill the array with random 0/1
    long long i = 0;
    for (i = 0; i < N; i++) {
        array[i] = rand() % 2;
    }

    // count all entries that hold '1'
    long long cnt = 0;
    for (i = 0; i < N; i++) {
        int entry = array[i];
        if (entry == 1)
            cnt++;
    }
    
    // print the result
    printf("Number of 1s: %lld\n", cnt);
    // free the allocated memory
    free(array);

    return EXIT_SUCCESS;
}
