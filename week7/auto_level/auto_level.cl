
__kernel void level(
    __global unsigned char* data,      // the vector to be reduced
    __global unsigned char* result,    // the result vector
    __local  unsigned char* scratch,   // a local scratch memory buffer (= W * E)
    long N,                  // the length of the data-vector 
    long W,                  // work_group_size ( = 3)
    long E                   // elements_to_check ( = 10)
) {

    // get Ids
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    // load data into local memory
    if (global_index < N) {
        scratch[local_index] = data[global_index];
    } else {
        scratch[local_index] = data[N-1];       // last element in data-array
    }
    
    // wait for all in group to flush results to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // perform reduction    
    if (local_index < W) {
		unsigned char min = scratch[local_index];
		unsigned char max = scratch[local_index + W];
		
		//swap if min > max
		if (min > max) {
			unsigned char tmp = min;
			min = max;
			max = tmp;
			scratch[local_index] = min;
			scratch[local_index + W] = max;
		}
		for (int i = 0; i < E - 2; i++) {
			if (scratch[local_index + ((i+2)*W)] < scratch[local_index]) scratch[local_index] = scratch[local_index + ((i+2)*W)];
			if (scratch[local_index + ((i+2)*W)] > scratch[local_index + W]) scratch[local_index + W] = scratch[local_index + ((i+2)*W)];
		}
		// sync on local memory state
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // write result to global result buffer
    result[get_group_id(0)] = scratch[local_index];
    result[get_group_id(0) + W] = scratch[local_index + W];
    
}

