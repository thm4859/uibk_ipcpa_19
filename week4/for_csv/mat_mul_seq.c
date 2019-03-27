#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

typedef float value_t;


// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

// ----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    int N = 1000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    // ---------- setup ----------

    // create two input matrices (on heap!)
    Matrix A = createMatrix(N,N);
    Matrix B = createMatrix(N,N);
    
    // fill matrices
    for(int i = 0; i<N; i++) {
        for(int j = 0; j<N; j++) {
            A[i*N+j] = i*j;             // some arbitrary matrix - note: flattend indexing!
            B[i*N+j] = (i==j) ? 1 : 0;  // identity
        }
    }
    
    // ---------- compute ----------
    
    Matrix C = createMatrix(N,N);

    timestamp begin = now();
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            value_t sum = 0;
            for(long long k = 0; k<N; k++) {
                sum += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
    timestamp end = now();

    // ---------- check ----------    
    //calculation: upper bound of FLOP: each integer operation as slow as a float one
    // also assuming the compiler does not just use inc and a cached variable to cut those operations down
    // so N*N*N(1+1+3+3) -> N³ for the loops then 1. for the += then 1 for the * in the middle (arguably the
    // only floating point operaton) finally +3 and +3 for the index calculation
    // making this probably a very wide upper bound
    bool success = true;
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            if (C[i*N+j] == i*j) continue;
            success = false;
            break;
        }
    }

    // ---------- cleanup ----------
    float mflops=((float) N)/1000.f; 
    mflops=mflops*8*(float)N*N/1000.f;
    releaseMatrix(A);
    releaseMatrix(B);
    releaseMatrix(C);
        printf("%d,", N);
    printf("%.3f,", (end-begin)*1000);
    if(success){
        printf("1,");
    }else{
        printf("0,");
    }
    printf("%d,", 1);
    // done
    printf("%.3f, \n",mflops/(end-begin));
    // done
    return (success) ? EXIT_SUCCESS : EXIT_FAILURE;
}


Matrix createMatrix(int N, int M) {
    // create data and index vector
    return malloc(sizeof(value_t)*N*M);
}

void releaseMatrix(Matrix m) {
    free(m);
}

