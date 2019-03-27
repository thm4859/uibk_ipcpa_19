#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"

typedef float value_t;


// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

// ----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    int N = 1000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    printf("Computing matrix-matrix product with N=%d\n", N);

    
    // ---------- setup ----------

    // create two input matrices (on heap!)
    Matrix A = createMatrix(N,N);
    Matrix B = createMatrix(N,N);
    
    // fill matrices
    for(int i = 0; i<N; i++) {
        for(int j = 0; j<N; j++) {
            A[i*N+j] = i*j;             // some arbitrary matrix - note: flattend indexing!
            B[i*N+j] = (i==j) ? 1 : 0;  // identity
        }
    }
    
    // ---------- compute ----------
    
    Matrix C = createMatrix(N,N);

    timestamp begin = now();
    
    {
        // -- solution with CL utils --

        // Part 1: ocl initialization
/*
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue);
*/        

    
		cl_int err;
		int i, j;
		char* value;
		size_t valueSize;
		cl_uint platformCount;
		cl_platform_id* platforms;
		cl_uint deviceCount;
		cl_device_id* devices;
		cl_uint maxComputeUnits;

		// get all platforms
		err = clGetPlatformIDs(0, NULL, &platformCount);
		CLU_ERRCHECK(err, "Failed to get clGetPlatformIDs");
		platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
		err = clGetPlatformIDs(platformCount, platforms, NULL);
		CLU_ERRCHECK(err, "Failed to get clGetPlatformIDs");

		for (i = 0; i < platformCount; i++) {

			// get all devices
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
			CLU_ERRCHECK(err, "Failed to get clGetDeviceIDs: CL_DEVICE_TYPE_ALL");
			//CLU_ERRCHECK(err, "Failed to write matrix B to device");
			devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
			CLU_ERRCHECK(err, "Failed to get clGetDeviceIDs: CL_DEVICE_TYPE_ALL");

			// for each device print critical attributes
			for (j = 0; j < deviceCount; j++) {

				// print device name
				err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DEVICE_NAME");
				value = (char*) malloc(valueSize);
				err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DEVICE_NAME");
				printf("%d. Device: %s\n", j+1, value);
				free(value);

				// print hardware device version
				err = clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DEVICE_VERSION");
				value = (char*) malloc(valueSize);
				err = clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DEVICE_VERSION");
				printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
				free(value);

				// print software driver version
				err = clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DRIVER_VERSION");
				value = (char*) malloc(valueSize);
				err = clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DRIVER_VERSION");
				printf(" %d.%d Software version: %s\n", j+1, 2, value);
				free(value);

				// print c version supported by compiler for device
				err = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DEVICE_OPENCL_C_VERSION");
				value = (char*) malloc(valueSize);
				err = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DEVICE_OPENCL_C_VERSION");
				printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
				free(value);

				// print parallel compute units
				clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
						sizeof(maxComputeUnits), &maxComputeUnits, NULL);
				CLU_ERRCHECK(err, "Failed to get clGetDeviceInfo: CL_DEVICE_MAX_COMPUTE_UNITS");
				printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

			}
			free(devices);
		}
		free(platforms);       
       
       
       cl_int  ciErrNum;
       cl_context context = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU,NULL, NULL, &ciErrNum);
       //cl_event prof_event;
       cl_command_queue command_queue; 
       cl_device_id device_id = cluInitDevice(0, &context, &command_queue);
       command_queue= clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
       
       
       


        // Part 2: create memory buffers
        cl_mem devMatA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, N * N * sizeof(value_t), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for matrix A");
        cl_mem devMatB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, N * N * sizeof(value_t), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for matrix B");
        cl_mem devMatC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, N * N * sizeof(value_t), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for matrix C");

        // Part 3: fill memory buffers
        err = clEnqueueWriteBuffer(command_queue, devMatA, CL_FALSE, 0, N * N * sizeof(value_t), A, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write matrix A to device");
        err = clEnqueueWriteBuffer(command_queue, devMatB, CL_TRUE, 0,  N * N * sizeof(value_t), B, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write matrix B to device");

        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "mat_mul.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "mat_mul", &err);
        CLU_ERRCHECK(err, "Failed to create mat_mul kernel from program");

        // Part 5: set arguments and execute kernel
        size_t size[2] = {N, N}; // two dimensional range
        cluSetKernelArguments(kernel, 4,
            sizeof(cl_mem), (void *)&devMatC,
            sizeof(cl_mem), (void *)&devMatA,
            sizeof(cl_mem), (void *)&devMatB,
            sizeof(int), &N
        );
        cl_event time_event;
        
        timestamp begin_clEnqueueNDRangeKernel = now();
        
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, size, NULL, 0, NULL, &time_event), "Failed to enqueue 2D kernel");
        
        clFinish(command_queue);
        err = clWaitForEvents(1, &time_event);
        cl_ulong start_time, end_time;
        double run_time;
        size_t return_bytes;
        
        err = clGetEventProfilingInfo(time_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &start_time, &return_bytes);
        CLU_ERRCHECK(err, "Failed to clGetEventProfilingInfo: CL_PROFILING_COMMAND_QUEUED");
        
        err = clGetEventProfilingInfo(time_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
        CLU_ERRCHECK(err, "Failed to clGetEventProfilingInfo: CL_PROFILING_COMMAND_END");
        
        run_time = (double)(end_time-start_time);
        printf("Kernel runtime: \t\t\t%.3f ms\n", run_time/1000000);
        
        

        // Part 6: copy results back to host
        err = clEnqueueReadBuffer(command_queue, devMatC, CL_TRUE, 0, N * N * sizeof(value_t), C, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed reading back result");
		
		timestamp end_clEnqueueReadBuffer = now();
		printf("Datatransfer time: \t\t\t%.3f ms\n", ((end_clEnqueueReadBuffer-begin_clEnqueueNDRangeKernel)*1000) - run_time/1000000);
		printf("Datatransfer time and kernelruntime: \t%.3f ms\n", (end_clEnqueueReadBuffer-begin_clEnqueueNDRangeKernel)*1000);
		
		
        // Part 7: cleanup
        // wait for completed operations (there should be none)
        CLU_ERRCHECK(clFlush(command_queue),    "Failed to flush command queue");
        CLU_ERRCHECK(clFinish(command_queue),   "Failed to wait for command queue completion");
        CLU_ERRCHECK(clReleaseKernel(kernel),   "Failed to release kernel");
        CLU_ERRCHECK(clReleaseProgram(program), "Failed to release program");

        // free device memory
        CLU_ERRCHECK(clReleaseMemObject(devMatA), "Failed to release Matrix A");
        CLU_ERRCHECK(clReleaseMemObject(devMatB), "Failed to release Matrix B");
        CLU_ERRCHECK(clReleaseMemObject(devMatC), "Failed to release Matrix C");

        // free management resources
        CLU_ERRCHECK(clReleaseCommandQueue(command_queue), "Failed to release command queue");
        CLU_ERRCHECK(clReleaseContext(context),            "Failed to release OpenCL context");
    }
    
    timestamp end = now();
    printf("Total time: \t\t\t\t%.3f ms\n", (end-begin)*1000);

    // ---------- check ----------    
    
    bool success = true;
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            if (C[i*N+j] == i*j) continue;
            success = false;
            break;
        }
    }
    
    printf("Verification: %s\n", (success)?"OK":"FAILED");
    
    // ---------- cleanup ----------
    
    releaseMatrix(A);
    releaseMatrix(B);
    releaseMatrix(C);
    
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

