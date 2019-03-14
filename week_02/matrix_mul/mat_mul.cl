
__kernel void mat_mul(
    __global float* C, 
    __global const float* A, 
    __global const float* B,
    int M, //5
    int N, //4
    int O  //2
) {
    // obtain position of this 'thread'
    size_t m = get_global_id(0); //j
    size_t o = get_global_id(1); //i

    // if beyond boundaries => skip this one
    
    // compute C := A * B
    float sum = 0;
    for(int k = 0; k<N; k++) {
		sum += A[k*M+m] * B[o*N+k];
		
				
		//sum += A[o*N+k] * B[k*N+m];
        //sum += A[i*N+k] * B[j+k*N]; //OK
    }
    //C[i*N+j] = sum; //OK
    C[o*M+m] = sum; //OK
}
