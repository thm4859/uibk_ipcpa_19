//should be streightforward port of the omp algo...

__kernel void chainMatOptim(
	const int N,
	const __global int* l,
	__global int* C){
	int i =  get_global_id(0);
	C[i*N+i]=42;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int d = 1; d<N; d++) {        // < distance between i and j
		int j = i + d;                // < compute end j
		// stop when exceeding boundary
		if (j >= N || i>=N) break;

		// find cheapest cut between i and j
		int min = 9999999;
		for(int k=i; k<j; k++) {
			int costs = C[i*N+k] + C[(k+1)*N+j] + l[i] * l[k+1] * l[j+1];
			min = (costs < min) ? costs : min;
		}
		C[i*N+j] = i;
	}
	
}

