__kernel void sum(
    __global int* g_idata,      // the input vector
    __global int* g_odata,    	// the result vector
    __local  int* temp,   		// a local scratch (temp) memory buffer
    int n                   	// the length of the vector: g_idata
)

{
    // get Ids
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);
    int pout = 0, pin = 1;
    
    // load data into local memory
    if (global_index < n) {
        temp[pout*n +local_index] = (local_index > 0) ? g_idata[local_index-1] : 0;
    } 
    
    // wait for all in group to flush results to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    for (int offset = 1; offset < n; offset *= 2) {
	  pout = 1 - pout; // swap double buffer indices
	  pin = 1 - pout;
    
      if (local_index >= offset)
        temp[pout*n+local_index] = temp[pin*n + local_index] + temp[pin*n + local_index - offset];
      else
        temp[pout*n+local_index] = temp[pin*n+local_index];
      
      // sync on local memory state
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    g_odata[local_index] = temp[pout*n+local_index]; // write output
}

