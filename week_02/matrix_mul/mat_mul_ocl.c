#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "utils.h"

typedef float value_t;

// -- kernel code utils --

typedef struct kernel_code {
    const char* code;
    size_t size;
} kernel_code;

kernel_code loadCode(const char* filename);

void releaseCode(kernel_code code);

// -----------------------

// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int X, int Y);

void releaseMatrix(Matrix m);

void printMatrix(Matrix m, int X, int Y);

// ----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    int N = 1000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    int M = N;
    int O = N;
    printf("Computing matrix-matrix product with N=%d\n", N);

    
    // ---------- setup ----------

    // create two input matrices (on heap!)
    Matrix A = createMatrix(M,N);
    Matrix B = createMatrix(N,O);
    
    // fill matrix A MxN
    for(int i = 0; i<M; i++) {
        for(int j = 0; j<N; j++) {
            A[i*N+j] = i*j;             // some arbitrary matrix - note: flattend indexing!
            //B[i*N+j] = (i==j) ? 1 : 0;  // identity
        }
    }
 
    // fill matrix B NxO  
    for(int i = 0; i<N; i++) {
        for(int j = 0; j<O; j++) {
            //A[i*N+j] = i*j;             // some arbitrary matrix - note: flattend indexing!
            B[i*O+j] = (i==j) ? 1 : 0;  // identity
        }
    }
    
    // ---------- compute ----------
    
    Matrix C = createMatrix(M,O);
    
    // -- BEGIN ASSIGNMENT --
    
    // TODO: parallelize the following computation using OpenCL
    
    /*
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            value_t sum = 0;
            for(long long k = 0; k<N; k++) {
                sum += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
    */
    timestamp begin = now();
    
    // --- OpenCL part ---
    {
        // OpenCL reference pages:
        // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
        
        // some local state variables
        cl_platform_id platform_id = NULL;
        cl_device_id device_id = NULL;
        cl_context context = NULL;
        cl_command_queue command_queue = NULL;
        cl_program program = NULL;
        cl_kernel kernel = NULL;
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;
        cl_int ret;

        // TODO: all return codes should be checked!
    
        // Part A - resource management
    
        // 1) get platform
        if (clGetPlatformIDs(1, &platform_id, &ret_num_platforms) != CL_SUCCESS){
			printf("clGetPlatformIDs != CL_SUCCESS");
			return -1;
		}
        
        // 2) get device
        if (clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices) != CL_SUCCESS){
			printf("clGetDeviceIDs != CL_SUCCESS");
			return -1;
		}

        // 3) create context
        context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
        if(ret < 0)
			printf("clCreateContext != CL_SUCCESS");
        
        // 4) create command queue
        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
        if(ret < 0)
			printf("clCreateCommandQueue != CL_SUCCESS");



        // Part B - data management
        
        // 5) create memory buffers on device
        size_t mat_sizeA = sizeof(value_t) * M * N; 
        size_t mat_sizeB = sizeof(value_t) * N * O;        
        size_t mat_sizeC = sizeof(value_t) * M * O;         
        cl_mem devMatA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, mat_sizeA, NULL, &ret);
        if(ret < 0)
			printf("clCreateBuffer != CL_SUCCESS");
        cl_mem devMatB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, mat_sizeB, NULL, &ret);
        if(ret < 0)
			printf("clCreateBuffer != CL_SUCCESS");
        cl_mem devMatC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, mat_sizeC, NULL, &ret);
        if(ret < 0)
			printf("clCreateBuffer != CL_SUCCESS");


        // 6) transfer input data from host to device (synchronously)
        if(clEnqueueWriteBuffer(command_queue, devMatA, CL_TRUE, 0, mat_sizeA, &A[0], 0, NULL, NULL) != CL_SUCCESS){
			printf("clEnqueueWriteBuffer != CL_SUCCESS");
			return -1;
		}
        if(clEnqueueWriteBuffer(command_queue, devMatB, CL_TRUE, 0, mat_sizeB, &B[0], 0, NULL, NULL) != CL_SUCCESS){
			printf("clEnqueueWriteBuffer != CL_SUCCESS");
			return -1;
		}



        // Part C - computation

        // 6) load kernel code from file
        kernel_code code = loadCode("mat_mul.cl");
        
        // 7) compile kernel program from source
        program = clCreateProgramWithSource(context, 1, &code.code,
				                      (const size_t *)&code.size, &ret);
		if(ret < 0)
			printf("clCreateProgramWithSource != CL_SUCCESS");		                      

        // 8) build program (compile + link for device architecture)
        if(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL) != CL_SUCCESS){
			printf("clBuildProgram != CL_SUCCESS");
			return -1;
		}
        
        // report kernel build errors
        if (ret != CL_SUCCESS) {

            // create a temporary buffer for the message
            size_t size = 1<<20;    // 1MB
            char* msg = malloc(size);
            size_t msg_size;

            // retrieve the error message
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, size, msg, &msg_size);

            // print the error message
            printf("Build Error:\n%s",msg);
            exit(1);
        }

        // 9) create OpenCL kernel
        kernel = clCreateKernel(program, "mat_mul", &ret);
        if(ret < 0)
			printf("clCreateKernel != CL_SUCCESS");

        // 10) set arguments
        if(clSetKernelArg(kernel, 0, sizeof(cl_mem), &devMatC) != CL_SUCCESS){
			printf("clSetKernelArg != CL_SUCCESS");
			return -1;
		}
        if(clSetKernelArg(kernel, 1, sizeof(cl_mem), &devMatA) != CL_SUCCESS){
			printf("clSetKernelArg != CL_SUCCESS");
			return -1;
		}
        if(clSetKernelArg(kernel, 2, sizeof(cl_mem), &devMatB) != CL_SUCCESS){
			printf("clSetKernelArg != CL_SUCCESS");
			return -1;
		}
        if(clSetKernelArg(kernel, 3, sizeof(int), &M) != CL_SUCCESS){
			printf("clSetKernelArg != CL_SUCCESS");
			return -1;
		}
        if(clSetKernelArg(kernel, 4, sizeof(int), &N) != CL_SUCCESS){
			printf("clSetKernelArg != CL_SUCCESS");
			return -1;
		}
        if(clSetKernelArg(kernel, 5, sizeof(int), &O) != CL_SUCCESS){
			printf("clSetKernelArg != CL_SUCCESS");
			return -1;
		}
                        
        // 11) schedule kernel
        //size_t global_work_offset = NULL;
		size_t global_work_size[2] = {M, O}; // M O 
		
        if(clEnqueueNDRangeKernel(command_queue, kernel, 
                    2, NULL, global_work_size, NULL, 
                    0, NULL, NULL
		) != CL_SUCCESS){
			printf("clEnqueueNDRangeKernel != CL_SUCCESS");
			return -1;
		}
        
        // 12) transfer data back to host
        if(clEnqueueReadBuffer(command_queue, devMatC, CL_TRUE, 0, mat_sizeC, &C[0], 0, NULL, NULL) != CL_SUCCESS){
			printf("clEnqueueReadBuffer != CL_SUCCESS");
			return -1;
		}
        
        // Part D - cleanup
        
        // wait for completed operations (should all have finished already)
        if(clFlush(command_queue) != CL_SUCCESS){
			printf("clFlush != CL_SUCCESS");
			return -1;
		}
        if(clFinish(command_queue) != CL_SUCCESS){
			printf("clFinish != CL_SUCCESS");
			return -1;
		}
        if(clReleaseKernel(kernel) != CL_SUCCESS){
			printf("clReleaseKernel != CL_SUCCESS");
			return -1;
		}
        if(clReleaseProgram(program) != CL_SUCCESS){
			printf("clReleaseProgram != CL_SUCCESS");
			return -1;
		}
        
        // free device memory
        if(clReleaseMemObject(devMatA) != CL_SUCCESS){
			printf("clReleaseMemObject != CL_SUCCESS");
			return -1;
		}
        if(clReleaseMemObject(devMatB) != CL_SUCCESS){
			printf("clReleaseMemObject != CL_SUCCESS");
			return -1;
		}
        if(clReleaseMemObject(devMatC) != CL_SUCCESS){
			printf("clReleaseMemObject != CL_SUCCESS");
			return -1;
		}
        
        // free management resources
        if(clReleaseCommandQueue(command_queue) != CL_SUCCESS){
			printf("clReleaseCommandQueue != CL_SUCCESS");
			return -1;
		}
        if(clReleaseContext(context) != CL_SUCCESS){
			printf("clReleaseContext != CL_SUCCESS");
			return -1;
		}

    }
    
    // -- END ASSIGNMENT --  
    
    
    timestamp end = now();
    printf("Total time: %.3f ms\n", (end-begin)*1000);

    // ---------- check ----------    
    //printMatrix(A, M, N);
    //printMatrix(B, N, O);
    //printMatrix(C, M, O);
    
    bool success = true;
    for(long long i = 0; i<M; i++) {
        for(long long j = 0; j<O; j++) {
            if (C[i*O+j] == i*j) continue;
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


Matrix createMatrix(int X, int Y) {
    // create data and index vector
    return malloc(sizeof(value_t)*X*Y);
}

void releaseMatrix(Matrix m) {
    free(m);
}

kernel_code loadCode(const char* filename) {
    size_t MAX_SOURCE_SIZE = 0x100000;

    FILE* fp;

    // load the source code containing the kernel
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel from file %s\n", filename);
        exit(1);
    }
    
    kernel_code res;
    res.code = (char*)malloc(MAX_SOURCE_SIZE);
    res.size = fread((char*)res.code, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    return res;
}

void releaseCode(kernel_code code) {
    free((char*)code.code);
}

// print matrix
void printMatrix(Matrix m, int X, int Y) {
	printf("M = %d\nO = %d\n", X, Y);
	for(long long i = 0; i<X; i++) {
		for(long long j = 0; j<Y; j++) {
			printf("%f ", m[i*Y+j]);
		}
		printf("\n");
	}
}
