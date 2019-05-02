
__kernel void level(
    __global ulong* data,      // the vector to be reduced
    __global ulong* result,    // the result vector
    __local  ulong* scratch,   // a local scratch memory buffer (= W * E)
    long N,                  // the length of the data-vector 
    long C,                  // components ( = 3)
    long E                   // elements_to_check ( = 10)
) {

    // get Ids
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    // load data into local memory
    if (global_index < N) {
        scratch[local_index] = data[global_index];
    } else {
        scratch[local_index] = data[999];       // last element in data-array
    }
    
    // wait for all in group to flush results to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // perform reduction    
    if (local_index < C && scratch[local_index + C] != 999) {
		unsigned char min = scratch[local_index];
		unsigned char max = scratch[local_index + C];
		
		//swap if min > max
		if (min > max) {
			unsigned char tmp = min;
			min = max;
			max = tmp;
			scratch[local_index] = min;
			scratch[local_index + C] = max;
		}
		for (int i = 0; i < E - 2; i++) {
			if (scratch[local_index + ((i+2)*C)] < scratch[local_index]) scratch[local_index] = scratch[local_index + ((i+2)*C)];
			if (scratch[local_index + ((i+2)*C)] > scratch[local_index + C]) scratch[local_index + C] = scratch[local_index + ((i+2)*C)];
		}
    }
    // sync on local memory state
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // write result to global result buffer
    if (local_index < C) {
		result[(2*C*get_group_id(0))+local_index] = scratch[local_index];
		result[(2*C*get_group_id(0))+local_index + C] = scratch[local_index + C];
    }
}

