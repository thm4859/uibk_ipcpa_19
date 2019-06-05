//naive first kernel:
__kernel void mat_mul(
    __global float* c, 
    __global const float* a, 
    __global const float* b,
    int N
) {
    // obtain position of this 'thread'
    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    // if beyond boundaries => skip this one
    if (i >= N || j >= N) return;

    // compute C := A * B
    float sum = 0;
    for(int k = 0; k<N; k++) {
        sum += a[i*N+k] * b[k*N+j];
    }
    c[i*N+j] = sum;
}



//first added local memory use:
// however it already needs padding:
// chose to do this the weird way: have it actually copy from too small to padded
// and then have a Kernel that removes the zeros form the result:
#define PADDINGX 32
__kernel void paddingAddZeroes(const int P, 
                               const __global float* input,
                               const int P_XL,
                               __global float* output) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0);
    const int ty = get_group_id(1)*PADDINGX + get_local_id(1);
 
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
                                  __global float* output) {
    
    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); 
    const int ty = get_group_id(1)*PADDINGX + get_local_id(1); 


    // Only store the result if within P * P bounds
    if (tx < P && ty < P) {
        output[ty*P + tx] = input[ty*P_XL + tx];
    }
}


#define TS 32

__kernel void mat_mul_loc1(
			__global float* C,
                      const __global float* A,
                      const __global float* B,
			const int N
                      ) {
    
    // Thread identifiers
    const int M=N;
    const int K=N;
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
    if (col >= N || row >= N) return;
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[globalCol*M + globalRow] = acc;
}
#define BLOCK_SIZE 32
__kernel void matrixMul4(
__global float* C, __global float* A, __global float* B,
const int N)
{
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
a += aStep, b += bStep)
{
// Local memory array As used to store the sub-matrix of A
__local float As[BLOCK_SIZE][BLOCK_SIZE];
// Local memory array Bs used to store the sub-matrix of B
__local float Bs[BLOCK_SIZE][BLOCK_SIZE];
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
