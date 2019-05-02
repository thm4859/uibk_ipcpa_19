
__kernel void adjust(
    __global float* data,      // the vector to be reduced
    __global float* result,    // the result vector
    __global float* min_fac,                  // array with x elements (components) 
    __global float* max_fac,                  // array with x elements (components)
    __global float* avg_val,				  // array with average values of the components    
    __local  float* scratch,   				  // a local scratch memory buffer (= 30)
    long C,							 		  // # components
    long N											//dim of png-data array
) {

    // get Ids
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    // load data into local memory
    if (global_index < N) {
        scratch[local_index] = data[global_index];
    } else {
        scratch[local_index] = 0.0f;       // last element in data-array
    }
 
    // wait for all in group to flush results to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // perform adjustment
    int component = local_index % C;   

		float difference = (scratch[local_index] - avg_val[component]);
		difference *= (scratch[local_index] < avg_val[component]) ? min_fac[component] : max_fac[component];
		scratch[local_index] = (difference + avg_val[component]);
    
    // sync on local memory state
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // write result to global result buffer
	if(global_index < N) {
		result[global_index] = (unsigned char)scratch[local_index];
		//result[(get_group_id(0)*get_local_size(0))+ local_index] = (unsigned char) scratch[local_index];
	}

}

