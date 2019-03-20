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

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

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
	printf("%f",m[(N/4)+(N*N/4)+1]);
}


// ----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    int N = 500;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    printf("Computing matrix-matrix product with N=%d\n", N);

    
    // ---------- setup ----------

    // create two input matrices (on heap!)
    Matrix A = createMatrix(N,N);
    
    // fill matrices
    for(int i = 0; i<N; i++) {
        for(int j = 0; j<N; j++) {
            A[i*N+j] = 273;             // temperature is 0° C everywhere (273 K)
        }
    }

    // and there is a heat source in one corner
    int source_x = N/4;
    int source_y = N/4;
    A[source_x*N+source_y] = 273 + 60;
    
    // ---------- compute ----------
    
    Matrix C = createMatrix(N,N);
    
    // -- BEGIN ASSIGNMENT --

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
        ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        
        // 2) get device
        ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

        // 3) create context
        context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
        
        // 4) create command queue
        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);



        // Part B - data management
        
        // 5) create memory buffers on device
        size_t vec_size = sizeof(value_t) * N * N; // quadratic form
        cl_mem devVecA = clCreateBuffer(context, CL_MEM_READ_WRITE, vec_size, NULL, &ret);
        cl_mem devVecC = clCreateBuffer(context, CL_MEM_READ_WRITE, vec_size, NULL, &ret);


        // 6) transfer input data from host to device (synchronously)
        ret = clEnqueueWriteBuffer(command_queue, devVecA, CL_TRUE, 0, vec_size, &A[0], 0, NULL, NULL);



        // Part C - computation

        // 6) load kernel code from file
        kernel_code code = loadCode("heat_stencil.cl");
        
        // 7) compile kernel program from source
        program = clCreateProgramWithSource(context, 1, &code.code,
				                      (const size_t *)&code.size, &ret);

        // 8) build program (compile + link for device architecture)
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        
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
        kernel = clCreateKernel(program, "heat_stencil", &ret);
	size_t global_work_size[2] = {N, N};
        // 10) set arguments
	printTemperature( A,  N,  N);
	for(int i=0; i<N*100/2;i++){
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &devVecC);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devVecA);
		ret = clSetKernelArg(kernel, 2, sizeof(int), &N);

		// 11) schedule kernel
		// size_t global_work_offset = 0;
		
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 
		            2, NULL, global_work_size, NULL, 
		            0, NULL, NULL
		);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devVecC);
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &devVecA);

		// 11) schedule kernel
		// size_t global_work_offset = 0;
		
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 
		            2, NULL, global_work_size, NULL, 
		            0, NULL, NULL
		);


	}
        // 12) transfer data back to host
        ret = clEnqueueReadBuffer(command_queue, devVecC, CL_TRUE, 0, vec_size, &C[0], 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, devVecA, CL_TRUE, 0, vec_size, &A[0], 0, NULL, NULL);//one of those is a waste
	
        // Part D - cleanup
        
        // wait for completed operations (should all have finished already)
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        
        // free device memory
        ret = clReleaseMemObject(devVecA);
        ret = clReleaseMemObject(devVecC);
        
        // free management resources
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

    }
    timestamp end = now();
    printTemperature( A,  N,  N);
    // -- END ASSIGNMENT --  
    
    
    
    printf("Total time: %.3f ms\n", (end-begin)*1000);

    // ---------- check ----------    
    
    //printMatrix(C, N);
    
   bool success = true;
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            value_t temp = A[i*N+j];
            if (273 <= temp && temp <= 273+60) continue;
            success = false;
            break;
        }
    }
    
printf("Verification: %s\n", (success)?"OK":"FAILED");
    // ---------- cleanup ----------
    
    releaseMatrix(A);
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
void printMatrix(Matrix m, int N) {
	printf("N = %d\n", N);
	for(long long i = 0; i<N; i++) {
		for(long long j = 0; j<N; j++) {
			printf("%f ", m[i*N+j]);
		}
		printf("\n");
	}
}
