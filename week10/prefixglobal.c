#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"
#include "people.h"

long long roundUpToMultiple(long long N, long long B) {
    if ((N % B) == 0) return N;
    return N + (B - (N%B));
}


int main(int argc, char** argv) {

    // size of input-array
    size_t N = 2048;
    size_t N2 = 2048;
    if (argc > 1) {
        N = atol(argv[1]);
	N2 = atol(argv[1]);
    }
    printf("Computing Prefix sum implementation for multiple work groups (prefixglobal) of N=%ld values\n", N);

    
    // ---------- setup ----------

    // create a buffer for storing random values
    int* data = (int*)malloc(N*sizeof(int));
    person_t* liste;
    // generate a list of N persons
    generate_list(&liste, 42, N);
    print_list( liste, N);
    int output[N];
    person_t res[N];
    if (!data) {
        printf("Unable to allocate enough memory\n");
        return EXIT_FAILURE;
    }
    
    // initializing random value buffer
    for(int i=0; i<N; i++) {
        data[i] = rand() % 6;
        //data[i] = 1;
    }

    // ---------- compute ----------
    printf("Counting ...\n");
    int out[128];  
 
    timestamp begin = now();
    {
        // - setup -
    
        size_t work_group_size = N;
	size_t histo_global = 128;//MAX_AGE+1;
	if(work_group_size>=1024){
		work_group_size=128;
	}
	size_t x=(roundUpToMultiple(N,work_group_size))/work_group_size;
        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(1, &context, &command_queue);

        // Part 2: create memory buffers
        cl_int err;
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");

        cl_mem list_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, N *sizeof(person_t), NULL, &err);
        cl_mem list_res = clCreateBuffer(context, CL_MEM_READ_WRITE, N *sizeof(person_t), NULL, &err);
        cl_mem histogram = clCreateBuffer(context, CL_MEM_READ_WRITE, histo_global *sizeof(int), NULL, &err);

        cl_mem devDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, histo_global*sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");
	cl_mem wSum = clCreateBuffer(context, CL_MEM_READ_WRITE, (roundUpToMultiple(N,work_group_size))/work_group_size *sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");
	cl_mem wRes = clCreateBuffer(context, CL_MEM_READ_WRITE, (roundUpToMultiple(N,work_group_size))/work_group_size *sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");
        // Part 3: fill memory buffers (transferring A is enough, B can be anything)
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, N * sizeof(int), data, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(command_queue, list_buffer, CL_TRUE, 0, N * sizeof(person_t), liste, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");

		printf("test %i \n",(int)((roundUpToMultiple(N,work_group_size))/work_group_size));
        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "prefixglobal.cl", NULL);
        cl_kernel wskernel = clCreateKernel(program, "Workpresum", &err);
        cl_kernel hskernel = clCreateKernel(program, "histogram_primitiv", &err);
        CLU_ERRCHECK(err, "Failed to create reduction kernel from program");
        cl_kernel sskernel = clCreateKernel(program, "Sumpresum", &err);
        CLU_ERRCHECK(err, "Failed to create reduction kernel from program");
        cl_kernel skernel = clCreateKernel(program, "sumsum", &err);
        CLU_ERRCHECK(err, "Failed to create reduction kernel from program");
		cl_kernel ckernel = clCreateKernel(program, "copy", &err);
        CLU_ERRCHECK(err, "Failed to create reduction kernel from program");

        // Part 5: perform multi-step reduction
        CLU_ERRCHECK(clFinish(command_queue), "Failed to wait for command queue completion");
        timestamp begin_prefix_sum = now();
	N=128; 
        // perform one stage of the reduction
        size_t global_size = roundUpToMultiple(N/2,work_group_size);
        
        // for debugging:
        printf("N: %lu, Global: %lu, WorkGroup: %lu\n", N, global_size, work_group_size);
        
        //histogram kernel set
		clSetKernelArg(hskernel, 0, sizeof(cl_mem), &list_buffer);
        clSetKernelArg(hskernel, 1, sizeof(cl_mem), &histogram);
        clSetKernelArg(hskernel, 2, sizeof(int), &N2);
		CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, hskernel, 1, NULL, &histo_global, &histo_global, 0, NULL, NULL), "Failed to enqueue reduction kernel");    
		//-----
		
		//all 3 presum kernels set
        // update kernel parameters
        clSetKernelArg(sskernel, 0, sizeof(cl_mem), &histogram);
        clSetKernelArg(sskernel, 1, sizeof(cl_mem), &devDataB);
        clSetKernelArg(sskernel, 2, 2 * global_size * sizeof(int), NULL);
        clSetKernelArg(sskernel, 3, sizeof(int), &N);
		
	    // submit kernel
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, sskernel, 1, NULL, &global_size, &global_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");
//err = clEnqueueReadBuffer(command_queue, devDataB, CL_TRUE, 0, 128 *sizeof(int), &out, 0, NULL, NULL);
	//now pre-result is in devDataB and workgoup-max is in wSum
        clSetKernelArg(sskernel, 0, sizeof(cl_mem), &wSum);
        clSetKernelArg(sskernel, 1, sizeof(cl_mem), &wRes);
        clSetKernelArg(sskernel, 2, 2 * global_size * sizeof(int), NULL);
        clSetKernelArg(sskernel, 3, sizeof(int), &N);
	work_group_size=work_group_size*2;
        global_size = roundUpToMultiple(N,work_group_size);
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, sskernel, 1, NULL, &x, &x, 0, NULL, NULL), "Failed to enqueue reduction kernel");    
        clSetKernelArg(skernel, 0, sizeof(cl_mem), &devDataB);
        clSetKernelArg(skernel, 1, sizeof(cl_mem), &wRes);
        clSetKernelArg(skernel, 2, sizeof(int), &N);
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, skernel, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");    
        
		//now prefix sum of histogram in devDataB
		clSetKernelArg(ckernel, 0, sizeof(cl_mem), &list_buffer);
		clSetKernelArg(ckernel, 1, sizeof(cl_mem), &list_res);
        clSetKernelArg(ckernel, 2, sizeof(cl_mem), &devDataB);
        clSetKernelArg(ckernel, 3, sizeof(int), &N2);
		CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, ckernel, 1, NULL, &histo_global, &histo_global, 0, NULL, NULL), "Failed to enqueue reduction kernel");    
		//copy kernel set:
	err = clEnqueueReadBuffer(command_queue, list_res, CL_TRUE, 0, N2 *sizeof(person_t), &res, 0, NULL, NULL);
		
        //clFinish(command_queue);
        CLU_ERRCHECK(clFinish(command_queue), "Failed to wait for command queue completion");
        timestamp end_prefix_sum = now();
        printf("\tprefix_sum took: %.3f ms\n", (end_prefix_sum - begin_prefix_sum)*1000);

        // download result from device
        print_list(res, N2);
       
		
        CLU_ERRCHECK(err, "Failed to download result from device");
        


        // Part 7: cleanup
        // wait for completed operations (there should be none)
        CLU_ERRCHECK(clFlush(command_queue),    "Failed to flush command queue");
        CLU_ERRCHECK(clFinish(command_queue),   "Failed to wait for command queue completion");
        CLU_ERRCHECK(clReleaseKernel(wskernel),   "Failed to release kernel");
        CLU_ERRCHECK(clReleaseKernel(sskernel),   "Failed to release kernel");
        CLU_ERRCHECK(clReleaseKernel(skernel),   "Failed to release kernel");
        CLU_ERRCHECK(clReleaseKernel(ckernel),   "Failed to release kernel");
	CLU_ERRCHECK(clReleaseKernel(hskernel),"failed kernel histogram");
        CLU_ERRCHECK(clReleaseProgram(program), "Failed to release program");

        // free device memory
        CLU_ERRCHECK(clReleaseMemObject(devDataA), "Failed to release data buffer A");
        CLU_ERRCHECK(clReleaseMemObject(devDataB), "Failed to release data buffer B");
	CLU_ERRCHECK(clReleaseMemObject(list_buffer), "Failed to release data buffer ");
	CLU_ERRCHECK(clReleaseMemObject(list_res), "Failed to release data buffer ");
	CLU_ERRCHECK(clReleaseMemObject(histogram), "Failed to release data buffer ");
	CLU_ERRCHECK(clReleaseMemObject(wSum), "Failed to release data buffer ");
	CLU_ERRCHECK(clReleaseMemObject(wRes), "Failed to release data buffer ");
        // free management resources
        CLU_ERRCHECK(clReleaseCommandQueue(command_queue), "Failed to release command queue");
        CLU_ERRCHECK(clReleaseContext(context),            "Failed to release OpenCL context");
    }
    timestamp end = now();
    printf("\ttotal - took: %.3f ms\n", (end-begin)*1000);


    // -------- print result -------
    //printf("\n\t\tinput\toutput\n");
    //for (int i = 0; i < MAX_AGE; i++) {
//		printf("Number %d:\t %d\n", i + 1, out[i]);
//	}

	//tests if the output[] data are correct
	char true[]="true";
	char false[]="false";
    

    // ---------- cleanup ----------
    free(liste);
    printf("check: %s\n", check(data, output, N) == 1 ? true : false);
    free(data);
    
    // done
    return EXIT_SUCCESS;
}
