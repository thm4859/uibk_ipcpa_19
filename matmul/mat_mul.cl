
//first added local memory use:
// however it already needs padding:
// chose to do this the weird way: have it actually copy from too small to padded
// and then have a Kernel that removes the zeros form the result:
__kernel void paddingAddZeroes(const int P, 
                               const __global float* input,
                               const int P_XL,
                               __global float* output,
                               const int PADDING) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDING + get_local_id(0);
    const int ty = get_group_id(1)*PADDING + get_local_id(1);
 
    // Check whether we are within bounds of the XL matrix
    if (tx < P_XL && ty < P_XL) {
 
        // Copy the input or pad a zero
        float value;
        if (tx < P && ty < P) {
            value = input[ty*P + tx];
        }
        else {
            value = 0.0f;
        }
 
        // Store the result
        output[ty*P_XL + tx] = value;
    }
}

__kernel void paddingRemoveZeroes(const int P_XL, 
                                  const __global float* input,
                                  const int P, 
                                  __global float* output,
                                  const int PADDING) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDING + get_local_id(0); 
    const int ty = get_group_id(1)*PADDING + get_local_id(1); 


    // Only store the result if within P * P bounds
    if (tx < P && ty < P) {
        output[ty*P + tx] = input[ty*P_XL + tx];
    }
}


#define BLOCK_SIZE 32
__kernel void matrixMul4(
	__global float* C,
	__global float* A,
	__global float* B,
	const int N,
	__const int BLOCK_SIZE_)
{

	// Local memory array As used to store the sub-matrix of A
	__local float As[BLOCK_SIZE][BLOCK_SIZE];
	// Local memory array Bs used to store the sub-matrix of B
	__local float Bs[BLOCK_SIZE][BLOCK_SIZE];
	
	float Csub = 0;
	// Block index
	int bx = get_group_id(0);
	int by = get_group_id(1);
	// Thread index
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	// Index of the first sub-matrix of A processed by the block
	int aBegin = N * BLOCK_SIZE * by;
	// Index of the last sub-matrix of A processed by the block
	int aEnd= aBegin + N - 1;
	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;
	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;
	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * N;
	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
		a <= aEnd;
		a += aStep, b += bStep){
		
		// Load the matrices from global memory to local memory
		// each thread loads one element of each matrix
		As[ty][tx] = A[a + N * ty + tx];
		Bs[ty][tx] = B[b + N * ty + tx];
		// Synchronize to make sure the matrices are loaded
		barrier(CLK_LOCAL_MEM_FENCE);
		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
			for (int k = 0; k < BLOCK_SIZE; ++k) {
				Csub += As[ty][k] * Bs[k][tx];
			}
		// Synchronize to make sure that the preceding computation is
		// done before loading two new sub-matrices of A and B in the
		// next iteration
		barrier(CLK_LOCAL_MEM_FENCE);
	} // end for each sub-matrix
	// Write the block sub-matrix to device memory
	// each thread writes one element
	int c = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + N * ty + tx] = Csub;
}
// end kernel
