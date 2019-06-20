//should be streightforward port of the omp algo...

__kernel void chainMatOptim(
	const int N,
	const int B,
	const int bd,
	const __global int* l,
	__global int* C){
	int bi =  get_global_id(0);
	
	barrier(CLK_LOCAL_MEM_FENCE);
//	for(int bd = 0; bd<NB; bd++) { //parallelized by for loop in host
	//for(int bi=0; bi<NB-bd; bi++) { parallelized by global size -> todo: be sure that only NB-bd workers are spawned
	int bj = bi + bd;

	// get lower-left corner of current blocks
	int ci = (bi+1)*B-1;
	int cj = bj*B;

	// process current block in wave-front order
	int count = 0;
	for(int d=0; d<2*B-1;d++) {
		int li=(d >= B ? B-1 : d);
		int lj=(d < B ? 0 : d-B+1);
		for(;li>=0 && lj<B; lj++,li--) {

			// get coordinated in C
			int i = ci - li;
			int j = cj + lj;

			// check whether the current cell still of interest
			if (i > j || i >=N || j >=N ) continue;

			// for main diagonal
			if (i == j) {
				C[i*N+j] = 0;
				continue;
			}

			// find cheapest cut between i and j
			int min = INT_MAX;
			for(int k=i; k<j; k++) {
				int costs = C[i*N+k] + C[(k+1)*N+j] + l[i] * l[k+1] * l[j+1];
				min = (costs < min) ? costs : min;
			}
			C[i*N+j] = min;

		}
	}
}





//old kernel...

__kernel void chainMatOptim_old(
	const int N,
	const __global int* l,
	__global int* C){
	int i =  get_global_id(0);
	C[i*N+i]=0;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int d = 1; d<N; d++) {        // < distance between i and j
		int j = i + d;                // < compute end j
		// stop when exceeding boundary
		if (j >= N || i>=N) break;

		// find cheapest cut between i and j
		int min = INT_MAX;
		for(int k=i; k<j; k++) {
			int costs = C[i*N+k] + C[(k+1)*N+j] + l[i] * l[k+1] * l[j+1];
			min = (costs < min) ? costs : min;
		}
		C[i*N+j] = min;
	}
	
}
