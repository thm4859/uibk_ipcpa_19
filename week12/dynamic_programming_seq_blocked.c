#include <unistd.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

int main(int argc, char** argv) {

  int minSize = 10;
  int maxSize = 20;

  // read problem size
  int N = 2000;
  if (argc > 1) {
    N = atoi(argv[1]);
  }

  int S = N+1;
  printf("Computing minimum cost for multiplying %d matrices ...\n",N);

  // generate random matrix sizes
  srand(0);
  int* l = (int*)malloc(sizeof(int)*S);
  for(int i=0; i<S; i++) {
    l[i] = ((rand() /  (float)RAND_MAX) * (maxSize - minSize)) + minSize;
  }

  // compute minimum costs
  int* C = (int*)malloc(sizeof(int)*N*N);

  double start = now();

  // initialize solutions for costs of single matrix
#pragma omp parallel num_threads(N)
  {
	int id = omp_get_thread_num();
	C[id*N+id]=0;


  }


  // compute minimal costs for multiplying A_i x ... x A_j

for(int d = 1; d<N; d++) {        // < distance between i and j
#pragma omp parallel  shared(C)
#pragma omp parallel num_threads(N-d)
{
//    for(int i=0; i<N; i++) {        // < starting at each i
	int i = omp_get_thread_num();	
      int j = i + d;                // < compute end j

      // stop when exceeding boundary
      if (j < N){

      // find cheapest cut between i and j
      int min = INT_MAX;

      for(int k=i; k<j; k++) {
        if(i!=k&&j!=k+1){
	while(C[i*N+k]==0 && C[(k+1)*N+j]==0){
		printf("dumb wait of thread %d \n", i);
	   	sleep(0.010); 	
	}
	}
        int costs = C[i*N+k] + C[(k+1)*N+j] + l[i] * l[k+1] * l[j+1];
        min = (costs < min) ? costs : min;
      }
      C[i*N+j] = min;
    }
  }
}

  double end = now();

  printf("Minimal costs: %d FLOPS\n", C[0*N+N-1]);
  printf("Total time: %.3fs\n", (end-start));

  // clean
  free(C);
  free(l);

  // done
  return EXIT_SUCCESS;
}
