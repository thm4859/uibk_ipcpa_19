#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"


long long roundUpToMultiple(long long N, long long B) {
    if ((N % B) == 0) return N;
    return N + (B - (N%B));
}


int main(int argc, char** argv) {

    // size of input-array
    size_t N = 2048*2;
    if (argc > 1) {
        N = atol(argv[1]);
    }
    printf("Computing Prefix sum implementation for a single work group according to Hillis and Steele of N=%ld values\n", N);

    
    // ---------- setup ----------

    // create a buffer for storing random values
    int* data = (int*)malloc(N*sizeof(int));
    int output[N];

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
        
    timestamp begin = now();
    {
        // - setup -
        
        size_t work_group_size = N;

        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(1, &context, &command_queue);

        // Part 2: create memory buffers
        cl_int err;
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");

        cl_mem devDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, (roundUpToMultiple(N,work_group_size)) *sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");


        // Part 3: fill memory buffers (transferring A is enough, B can be anything)
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, N * sizeof(int), data, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");

        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "hillissteele.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "sum", &err);
        CLU_ERRCHECK(err, "Failed to create reduction kernel from program");

        // Part 5: perform multi-step reduction
        CLU_ERRCHECK(clFinish(command_queue), "Failed to wait for command queue completion");
        timestamp begin_prefix_sum = now();
        
        // perform one stage of the reduction
        size_t global_size = roundUpToMultiple(N,work_group_size);
        
        // for debugging:
        printf("N: %lu, Global: %lu, WorkGroup: %lu\n", N, global_size, work_group_size);
    
        // update kernel parameters
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &devDataA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &devDataB);
        clSetKernelArg(kernel, 2, 2 * global_size * sizeof(int), NULL);
        clSetKernelArg(kernel, 3, sizeof(int), &N);
	
	    // submit kernel
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");
            
        //clFinish(command_queue);
        CLU_ERRCHECK(clFinish(command_queue), "Failed to wait for command queue completion");
        timestamp end_prefix_sum = now();
        printf("\tprefix_sum took: %.3f ms\n", (end_prefix_sum - begin_prefix_sum)*1000);

        // download result from device
        err = clEnqueueReadBuffer(command_queue, devDataB, CL_TRUE, 0, N * sizeof(int), &output, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to download result from device");
        


        // Part 7: cleanup
        // wait for completed operations (there should be none)
        CLU_ERRCHECK(clFlush(command_queue),    "Failed to flush command queue");
        CLU_ERRCHECK(clFinish(command_queue),   "Failed to wait for command queue completion");
        CLU_ERRCHECK(clReleaseKernel(kernel),   "Failed to release kernel");
        CLU_ERRCHECK(clReleaseProgram(program), "Failed to release program");

        // free device memory
        CLU_ERRCHECK(clReleaseMemObject(devDataA), "Failed to release data buffer A");
        CLU_ERRCHECK(clReleaseMemObject(devDataB), "Failed to release data buffer B");

        // free management resources
        CLU_ERRCHECK(clReleaseCommandQueue(command_queue), "Failed to release command queue");
        CLU_ERRCHECK(clReleaseContext(context),            "Failed to release OpenCL context");
    }
    timestamp end = now();
    printf("\ttotal - took: %.3f ms\n", (end-begin)*1000);


    // -------- print result -------
//    printf("\n\t\tinput\toutput\n");
//    for (int i = 0; i < N; i++) {
//		printf("Number %d:\t%d\t%d\n", i + 1, data[i], output[i]);
//	}

	//tests if the output[] data are correct
	char true[]="true";
	char false[]="false";
    printf("check: %s\n", check(data, output, N) == 1 ? true : false);

    // ---------- cleanup ----------
    
    free(data);
    
    // done
    return EXIT_SUCCESS;
}

