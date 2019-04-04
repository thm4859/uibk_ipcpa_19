
typedef float value_t;

__kernel void stencil(
    __global const value_t* A, 
    __global value_t* B,
    __local value_t* localData,
    int source_x,
    int source_y,
    int N,
    int M
) {
    // obtain position of this 'thread'
    size_t i = get_global_id(0);
    size_t j = get_global_id(1) -1; // Shift offset up 1 radius (1 row) for reads
    size_t ii = get_local_id(0);
    size_t jj = get_local_id(1);

	int iDevGMEMOffset = mul24(j, (int) get_global_size(0)) + i; 
	
	// Compute initial offset of current pixel within work group LMEM block
	int iLocalPixOffset = mul24((int) get_local_id(1), M) + get_local_id(0)+1;

	// Main read of GMEM data into LMEM
	if((j > -1) && (j < N) && (i < N)) {
		localData[iLocalPixOffset] = A[iDevGMEMOffset];
	}else{
		localData[iLocalPixOffset] = (value_t)0;
	}


	// Work items with j ID < 2 read bottom 2 rows of LMEM
	if (get_local_id(1) < 2) {
		// Increase local offset by 1 workgroup LMEM block height
		// to read in top rows from the next block region down
		iLocalPixOffset += mul24((int)get_local_size(1), M);

		// If source offset is within the image boundaries
		if (((j + get_local_size(1)) < N) && (i < N)){
			// Read in top rows from the next block region down
			localData[iLocalPixOffset] = A[iDevGMEMOffset + mul24((int)get_local_size(1), (int)get_global_size(0))];
		}else{
			localData[iLocalPixOffset] = (value_t)0;
		}
	}

	
	// Work items with i ID at right workgroup edge will read left apron pixel
	if (get_local_id(0) == (get_local_size(0) - 1)){
		// MISSING
		return; // look in the source for details
	}

	// Work items with i ID at left workgroup edge will read right apron pixel
	else if (get_local_id(0) == 0){
		// MISSING
		return; // look in the source for details
	}
	
	// Synchronize the read into LMEM
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	// Here follows gradient computation on localData[]
	
    // center stays constant (the heat is still on)
    if (i == source_x && j == source_y) {
        B[i*N+j] = A[i*N+j];
        return;
    }

    // get current temperature at (ii,jj)
    value_t tc = localData[ii*N+jj];

    // get temperatures left/right and up/down
    value_t tl = ( jj !=  0  ) ? localData[ii*M+(jj-1)] : tc;
    value_t tr = ( jj != M-1 ) ? localData[ii*M+(jj+1)] : tc;
    value_t tu = ( ii !=  0  ) ? localData[(ii-1)*M+jj] : tc;
    value_t td = ( ii != M-1 ) ? localData[(ii+1)*M+jj] : tc;

    // update temperature at current point
    int corr = 0;
    if (jj >=  1) {
		corr = 1;
    }
    if(jj > 1){
		corr = 2;
    }
    B[iDevGMEMOffset] = tc + 0.2f * (tl + tr + tu + td + (-4.0f*tc));



	//flop calculation:
	
	//center: 4 (indices) -> 1 times on stencil
	//get temp left/right/up/down: 4 * 2 (indices) = 8
	//update temp: 7 + 2 (indices) = 9
	
	//flop for each part:
	//center: (T-1) * 4
	//get temp l/r/u/d: (T-1) * ((N * N * 8) - ((4 * N) + 4))       //((4 * N) + 4) = border
	//update temp: (T-1) * N * N * 9
	
	//comlplete flop calculation:
	//(((T-1) * 4) + ((T-1) * ((N * N * 8) - ((4 * N) + 4))) + ((T-1) * N * N * 9))
}
