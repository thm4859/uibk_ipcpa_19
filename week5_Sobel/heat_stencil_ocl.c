#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"


typedef float value_t;

// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

void printTemperature(Matrix m, int N, int M);

unsigned long long getElapsed(cl_event event);


// ----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    int N = 500;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    int T = N*100;
    cl_event events[T];
    
    printf("Computing heat-distribution for room size N=%d for T=%d timesteps\n", N, T);

    
    // ---------- setup ----------

    // create a buffer for storing temperature fields
    Matrix A = createMatrix(N,N);
    
    // set up initial conditions in A
    for(int i = 0; i<N; i++) {
        for(int j = 0; j<N; j++) {
            A[i*N+j] = 273;             // temperature is 0° C everywhere (273 K)
        }
    }

    // and there is a heat source in one corner
    int source_x = N/4;
    int source_y = N/4;
    A[source_x*N+source_y] = 273 + 60;

    printf("Initial:\n");
    printTemperature(A,N,N);
    
    // ---------- compute ----------


    unsigned long long all_events_run_kernel = 0.0f;

    timestamp begin = now();
    


    // -- BEGIN ASSIGNMENT --
    
    // - setup -

    // Part 1: ocl initialization
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device_id = cluInitDevice(0, &context, &command_queue);


    
    cl_int err;
    
    // for enabling: clGetEventProfilingInfo() -> CL_QUEUE_PROFILING_ENABLE
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    CLU_ERRCHECK(err, "Failed to enable CL_QUEUE_PROFILING_ENABLE on command queue");
    
    // Part 2: create memory buffers
    cl_mem devMatA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(value_t), NULL, &err);
    CLU_ERRCHECK(err, "Failed to create buffer for matrix A");
    cl_mem devMatB = clCreateBuffer(context, CL_MEM_READ_WRITE, N * N * sizeof(value_t), NULL, &err);
    CLU_ERRCHECK(err, "Failed to create buffer for matrix B");

    // Part 3: fill memory buffers (transfering A is enough, B can be anything)
    err = clEnqueueWriteBuffer(command_queue, devMatA, CL_TRUE, 0, N * N * sizeof(value_t), A, 0, NULL, NULL);
    CLU_ERRCHECK(err, "Failed to write matrix A to device");

    // Part 4: create kernel from source
    cl_program program = cluBuildProgramFromFile(context, device_id, "heat_stencil.cl", NULL);
    cl_kernel kernel = clCreateKernel(program, "stencil", &err);
    CLU_ERRCHECK(err, "Failed to create mat_mul kernel from program");

    // Part 5: set arguments in kernel (those which are constant)
    int iBlockDim = 3;
    clSetKernelArg(kernel, 3, sizeof(int), &source_x);
    clSetKernelArg(kernel, 4, sizeof(int), &source_y);
    clSetKernelArg(kernel, 5, sizeof(int), &N);
    clSetKernelArg(kernel, 6, sizeof(int), &iBlockDim+2);
    
    

    // for each time step ..
    bool dirty = false;
    for(int t=0; t<T; t++) {

		// mark host-side buffer dirty
        dirty = true;

        // enqeue a kernel call for the current time step
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &devMatA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &devMatB);
        clSetKernelArg(kernel, 2, ((iBlockDim + 2) * (iBlockDim + 2) * sizeof(value_t)), NULL);
        
        size_t size[2] = {N, N}; // two dimensional range
        size_t size_local[2] = {(iBlockDim+2), (iBlockDim+2)};
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, size, size_local, 0, NULL, &events[t]), "Failed to enqueue 2D kernel");

        // swap matrices (just handles, no conent)
        cl_mem tmp = devMatA;
        devMatA = devMatB;
        devMatB = tmp;

        // show intermediate step
        if (!(t%1000)) {

            // download state of A to host
            err = clEnqueueReadBuffer(command_queue, devMatA, CL_TRUE, 0, N * N * sizeof(value_t), A, 0, NULL, NULL);
            CLU_ERRCHECK(err, "Failed to read matrix A from device");

            // revert dirty flag
            dirty = false;

            // print the step
            printf("Step t=%d:\n", t);
            printTemperature(A,N,N);
        }		
        
    }

    // get back final version of A
    if (dirty) {
        // download state of A to host
        err = clEnqueueReadBuffer(command_queue, devMatA, CL_TRUE, 0, N * N * sizeof(value_t), A, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to read matrix A from device");
    }

    // Part 7: cleanup
    // wait for completed operations (there should be none)
    CLU_ERRCHECK(clFlush(command_queue),    "Failed to flush command queue");
    CLU_ERRCHECK(clFinish(command_queue),   "Failed to wait for command queue completion");
    CLU_ERRCHECK(clReleaseKernel(kernel),   "Failed to release kernel");
    CLU_ERRCHECK(clReleaseProgram(program), "Failed to release program");

    // free device memory
    CLU_ERRCHECK(clReleaseMemObject(devMatA), "Failed to release Matrix A");
    CLU_ERRCHECK(clReleaseMemObject(devMatB), "Failed to release Matrix B");

    // free management resources
    CLU_ERRCHECK(clReleaseCommandQueue(command_queue), "Failed to release command queue");
    CLU_ERRCHECK(clReleaseContext(context),            "Failed to release OpenCL context");

    // -- END ASSIGNMENT --
    

    timestamp end = now();
    
   

    // ---------- check ----------    

    printf("Final:\n");
    printTemperature(A,N,N);
    
    bool success = true;
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            value_t temp = A[i*N+j];
            if (273 <= temp && temp <= 273+60) continue;
            success = false;
            break;
        }
    }
    
	
	// compute MFLOPs -> more information in "heat_stencil.cl"
	double num_mflop = (((T-1) * 4) + ((T-1) * ((N * N * 8) - ((4 * N) + 4))) + ((T-1) * N * N * 9))/1e6;	
	
	// calculate kernel time
	for (int i = 0; i < T; i++){
		all_events_run_kernel += getElapsed(events[i]);
	}
	
	// compute performnce of kernel
    printf("Total time: \t\t%.3f ms\n", (end-begin)*1000);
	printf("Kernel time:\t\t%.3f ms\n", (all_events_run_kernel/1e6));
	printf("Performance kernel:\t%.3f MFLOP/s\n", num_mflop/(all_events_run_kernel/1e9));    
    printf("Verification: %s\n", (success)?"OK":"FAILED");
    
    // ---------- cleanup ----------
    
    releaseMatrix(A);
    
    // done
    return (success) ? EXIT_SUCCESS : EXIT_FAILURE;
}



Matrix createMatrix(int N, int M) {
    // create data and index vector
    return malloc(sizeof(value_t)*N*M);
}

void releaseMatrix(Matrix m) {
    free(m);
}

void printTemperature(Matrix m, int N, int M) {
    const char* colors = " .-:=+*#%@";
    const int numColors = 10;

    // boundaries for temperature (for simplicity hard-coded)
    const value_t max = 273 + 30;
    const value_t min = 273 + 0;

    // set the 'render' resolution
    int H = 30;
    int W = 50;

    // step size in each dimension
    int sH = N/H;
    int sW = M/W;


    // upper wall
    for(int i=0; i<W+2; i++) {
        printf("X");
    }
    printf("\n");

    // room
    for(int i=0; i<H; i++) {
        // left wall
        printf("X");
        // actual room
        for(int j=0; j<W; j++) {

            // get max temperature in this tile
            value_t max_t = 0;
            for(int x=sH*i; x<sH*i+sH; x++) {
                for(int y=sW*j; y<sW*j+sW; y++) {
                    max_t = (max_t < m[x*N+y]) ? m[x*N+y] : max_t;
                }
            }
            value_t temp = max_t;

            // pick the 'color'
            int c = ((temp - min) / (max - min)) * numColors;
            c = (c >= numColors) ? numColors-1 : ((c < 0) ? 0 : c);

            // print the average temperature
            printf("%c",colors[c]);
        }
        // right wall
        printf("X\n");
    }

    // lower wall
    for(int i=0; i<W+2; i++) {
        printf("X");
    }
    printf("\n");

}

unsigned long long getElapsed(cl_event event) {
    cl_ulong starttime = 0, endtime = 0;
    CLU_ERRCHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL), "Failed to get profiling information");
    CLU_ERRCHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL), "Failed to get profiling information");
	return (endtime-(unsigned long long)starttime);
}
