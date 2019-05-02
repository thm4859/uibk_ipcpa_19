
__kernel void adjust(
    __global uchar* data,      // the vector to be reduced
    __global uchar* result,    // the result vector
    //__global float* min_fac,                  // array with x elements (components) 
    //__global float* max_fac,                  // array with x elements (components)
    //__global uchar* avg_val,				  // array with average values of the components    
    __local  uchar* scratch,   				  // a local scratch memory buffer (= 30)
    unsigned long C,
    float min1,
	float min2,
	float min3,
	float max1,
	float max2,
	float max3,
	unsigned char avg1,
	unsigned char avg2,
	unsigned char avg3 							 		  // # components
) {

    // get Ids
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    // load data into local memory
    scratch[local_index] = data[global_index];
 
    // wait for all in group to flush results to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // perform adjustment
    int component =  local_index % C;   
    if (component == 0) {
		//float difference = (float)(scratch[local_index] - avg_val[component]);
		//difference *= (difference < avg_val[component]) ? min_fac[component] : max_fac[component];
		//scratch[local_index] =  (uchar)(difference + avg_val[component]);
		float difference = (float)(scratch[local_index] - avg1);
		difference *= (difference < avg1) ? min1 : max1;
		scratch[local_index] =  (uchar)(difference + avg1);		
    }
    if (component == 1) {
		//float difference = (float)(scratch[local_index] - avg_val[component]);
		//difference *= (difference < avg_val[component]) ? min_fac[component] : max_fac[component];
		//scratch[local_index] =  (uchar)(difference + avg_val[component]);
		float difference = (float)(scratch[local_index] - avg2);
		difference *= (difference < avg2) ? min2 : max2;
		scratch[local_index] =  (uchar)(difference + avg2);		
    }
    if (component == 2) {
		//float difference = (float)(scratch[local_index] - avg_val[component]);
		//difference *= (difference < avg_val[component]) ? min_fac[component] : max_fac[component];
		//scratch[local_index] =  (uchar)(difference + avg_val[component]);
		float difference = (float)(scratch[local_index] - avg3);
		difference *= (difference < avg3) ? min3 : max3;
		scratch[local_index] =  (uchar)(difference + avg3);		
    }
    
    // sync on local memory state
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // write result to global result buffer
	result[(get_group_id(0)*get_local_size(0))+ local_index] = scratch[local_index];

}

