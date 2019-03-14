
__kernel void mat_mul(
    __global float* C, 
    __global const float* A, 
    __global const float* B,
    int M, //row A, row C
    int N, //column A, row B
    int O  //column B, column C
) {
    // obtain position of this 'thread'
    size_t m = get_global_id(0);
    size_t o = get_global_id(1);

    // if beyond boundaries => skip this one
    
    // compute C := A * B
    float sum = 0;
    for(int k = 0; k<N; k++) {
		sum += A[m*N+k] * B[O*k+o];	
    }
    C[m*O+o] = sum;
}
