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
        int global_index = get_global_id(0);
    int local_index = get_local_id(0);
    int offset = 1;
    int group_size=(n/get_num_groups(0));
    int group_off=(get_group_id(0)*group_size);
    //if(group_off==0){return;}
    // load data into local memory
    temp[2*local_index] = g_idata[(2*global_index)]; // load input into shared memory
    temp[2*local_index+1] = g_idata[(2*global_index+1)];
    
    // wait for all in group to flush results to local memory
    //barrier(CLK_LOCAL_MEM_FENCE);
    
    
    for (int d = (n/get_num_groups(0))>>1; d > 0; d >>= 1) { // build sum in place up the tree
		barrier(CLK_LOCAL_MEM_FENCE);
		if (local_index < d) {
			int ai = offset*(2*local_index+1)-1;
			int bi = offset*(2*local_index+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
    
    if (local_index == 0) { 
	w_sum[get_group_id(0)]=temp[(n/get_num_groups(0)) - 1];
	temp[(n/get_num_groups(0)) - 1] = 0; 
    } // clear the last element
	for (int d = 1; d < (n/get_num_groups(0)); d *= 2) { // traverse down tree & build scan
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
    
    if(2*global_index<(get_group_id(0)+1)*group_size && 2*global_index>(get_group_id(0))*group_size){
    	g_odata[2*global_index] =temp[2*local_index]; // write results to device memory
	}
    if(2*global_index+1<(get_group_id(0)+1)*group_size && 2*global_index+1>(get_group_id(0))*group_size){
    	g_odata[2*global_index+1] = temp[2*local_index+1];
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
    temp[2*local_index] = g_idata[2*local_index]; // load input into shared memory
    temp[2*local_index+1] = g_idata[2*local_index+1];
    
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
    
    if (local_index == 0) { temp[n - 1] = 0; } // clear the last element
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
    
    
    g_odata[2*global_index] = temp[2*local_index]; // write results to device memory
    g_odata[2*global_index+1] = temp[2*local_index+1];
}

__kernel void sumsum(
    __global int* g_data,      // the input vector
    __global int* sum_data,    	// the result vector
     int n                   	// the length of the vector: g_idata
)
{
    int global_index = get_global_id(0);
    int w_ind = get_group_id(0);

	if(n>global_index){
		g_data[global_index]=g_data[global_index]+sum_data[w_ind];
	}else{return;}

}
#define MAX_AGE 120
#define NAME_LEN 32

typedef char name_t[NAME_LEN];

typedef struct __attribute__ ((packed)){
	int age;
	name_t name;
} person_t;


__kernel void histogram_primitiv(
	__global person_t* liste,
	__global int* histy,
	int n
){
//parallelisation strat for this kernel:
//have n batches, for n workgroups calculating n histogramms (< 128 histograms no doubt)
//and have then a reduction step on these -> 1 step at most
//all in all probably same or greater number of global memory access so not worth the logic overhead
	int global_index = get_global_id(0);
	int b =0;
	
	for(int i=0;i<n;i++){
		if(liste[i].age==global_index){
			b=b+1;
		}
	}
	histy[global_index]=b;

}

__kernel void copy(
	__global person_t* start,
	__global person_t* end,
	__global int* histy,
    int n
//128 worker die alle die schleife durchgehen und einfügen
// use of local memory is possible:
//have a local_memsize sized batch of start data and then load them into memory
//(in parallell by the 128(or N superfluous) workers) and loop through the batches of start
//overhead in opencl host logic so ommitted for now
){
	int global_index = get_global_id(0);

	int own_offset=0;
	for(int i=0;i<n;i++){
		if(start[i].age==global_index){
			end[histy[global_index]+own_offset]=start[i];
			own_offset=own_offset+1;
		}
	}

}
