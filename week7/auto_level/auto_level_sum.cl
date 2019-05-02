
__kernel void sum(
    __global ulong* data,      // the vector to be reduced
    __global ulong* result,    // the result vector
    __local  ulong* scratch,   // a local scratch memory buffer (= C * E)
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
        scratch[local_index] = 999;       // last element in data-array
    }
    
    // wait for all in group to flush results to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // perform reduction    
    if (local_index < C) {
		for (int i = 1; i < E; i++) {
			if (scratch[local_index + ((i)*C)] != 999){
				scratch[local_index] += scratch[local_index + ((i)*C)];
			}
		}
		
    }
    // sync on local memory state
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // write result to global result buffer
    if (local_index < C) {
		result[(get_group_id(0)*C)+ local_index] = scratch[local_index];
    }
}

