#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__kernel void Workpresum(
    __global int* g_idata,      // the input vector
    __global int* g_odata,    	// the result vector
    __global int* w_sum,
    __local  int* temp,   		// a local scratch (temp) memory buffer
    int n                   	// the length of the vector: g_idata
)

{
    // get Ids
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);
    int offset = 1;
    
    // load data into local memory
//A    temp[2*local_index] = g_idata[2*global_index]; // load input into shared memory
//A    temp[2*local_index+1] = g_idata[2*global_index+1];
    
//Block A    
    int ai = local_index;
	int bi = local_index + (n/2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
    
    
    
    // wait for all in group to flush results to local memory
    //barrier(CLK_LOCAL_MEM_FENCE);
    
    
    for (int d = n>>1; d > 0; d >>= 1) { // build sum in place up the tree
		barrier(CLK_LOCAL_MEM_FENCE);
		if (local_index < d) {
			int ai = offset*(2*local_index+1)-1;
			int bi = offset*(2*local_index+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
    
//C    if (local_index == 0) { temp[n - 1] = 0; } // clear the last element
    if (local_index==0) { 
	
	temp[n-1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; 

	}
    
	for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (global_index < d) {
			int ai = offset*(2*global_index+1)-1;
			int bi = offset*(2*global_index+2)-1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

      // sync on local memory state
      barrier(CLK_LOCAL_MEM_FENCE);
    
    
//E    g_odata[2*global_index] = temp[2*local_index]; // write results to device memory
//E    g_odata[2*global_index+1] = temp[2*local_index+1];

    g_odata[ai] = temp[ai + bankOffsetA]+1;
	g_odata[bi] = temp[bi + bankOffsetB]+1;
	if(global_index%(n/get_num_groups(0))==0){
		w_sum[get_group_id(0)]=g_odata[global_index];
	}

}

__kernel void Sumpresum(
    __global int* g_idata,      // the input vector
    __global int* g_odata,    	// the result vector
    __local  int* temp,   		// a local scratch (temp) memory buffer
    int n                   	// the length of the vector: g_idata
)

{

    // get Ids
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);
    int offset = 1;
    
    // load data into local memory
//A    temp[2*local_index] = g_idata[2*global_index]; // load input into shared memory
//A    temp[2*local_index+1] = g_idata[2*global_index+1];
    
//Block A    
    int ai = local_index;
	int bi = local_index + (n/2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
    
    
    
    // wait for all in group to flush results to local memory
    //barrier(CLK_LOCAL_MEM_FENCE);
    
    
    for (int d = n>>1; d > 0; d >>= 1) { // build sum in place up the tree
		barrier(CLK_LOCAL_MEM_FENCE);
		if (local_index < d) {
			int ai = offset*(2*local_index+1)-1;
			int bi = offset*(2*local_index+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
    
//C    if (local_index == 0) { temp[n - 1] = 0; } // clear the last element
    if (local_index==0) { 
	temp[n-1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; 

	}
    
	for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (local_index < d) {
			int ai = offset*(2*local_index+1)-1;
			int bi = offset*(2*local_index+2)-1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

      // sync on local memory state
      barrier(CLK_LOCAL_MEM_FENCE);
    
    
//E    g_odata[2*global_index] = temp[2*local_index]; // write results to device memory
//E    g_odata[2*global_index+1] = temp[2*local_index+1];
    g_odata[ai] = temp[ai + bankOffsetA]+1;
	g_odata[bi] = temp[bi + bankOffsetB]+1;
}

__kernel void sumsum(
    __global int* g_data,      // the input vector
    __global int* sum_data,    	// the result vector
     int n                   	// the length of the vector: g_idata
)
{
    int global_index = get_global_id(0);
    int w_ind = get_group_id(0);
    
	if(w_ind==0){
		return;//nothing to do
	}
	if(n>global_index){
		g_data[global_index]=g_data[global_index]+sum_data[w_ind-1];
	}else{return;}

}
