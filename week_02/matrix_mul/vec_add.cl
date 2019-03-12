
__kernel void vec_add(
    __global float* C, 
    __global const float* A, 
    __global const float* B,
    int N,
    int M
) {
    // obtain position of this 'thread'
    size_t j = get_global_id(0);
    size_t i = get_global_id(1);

    // if beyond boundaries => skip this one
    
    // compute C := A + B
    float sum = 0;
    for(int k = 0; k<N; k++) {
        sum += A[i*N+k] * B[k*N+j];
    }
    C[i*N+j] = sum;
}
