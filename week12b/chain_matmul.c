#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "cl_utils.h"

typedef float value_t;


// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

// ----------------------

typedef struct _cl_mm_environment {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel chain;//have a better name than kernel!
} cl_mm_environment;

cl_mm_environment createMMEnvironment();

void destroyMMEnvironment(cl_mm_environment);

int roundUpToMultiple(int N, int B) {
    if ((N % B) == 0) return N;
    return N + (B - (N%B));
}



int main(int argc, char** argv) {


    // ---------- setup ----------

    cl_mm_environment env = createMMEnvironment();
	int minSize = 10;
	int maxSize = 20;
	int N = 20;
	if (argc > 1) {
		N = atoi(argv[1]);
	}
    
    

	int S = N+1;
	printf("Computing minimum cost for multiplying %d matrices ...\n",N);

	// generate random matrix sizes
	srand(0);
	int* l = (int*)malloc(sizeof(int)*S);
	for(int i=0; i<S; i++) {
		l[i] = ((rand() /  (float)RAND_MAX) * (maxSize - minSize)) + minSize;
	}

	// compute minimum costs
	int* C = (int*)malloc(sizeof(int)*N*N);
	int* D = (int*)malloc(sizeof(int)*N*N);
	

        // fill matrix
        for(int i = 0; i<N; i++) {
            for(int j = 0; j<N; j++) {
                C[i*N+j] = 0.0; 
                D[i*N+j] = 0.0;
            }
        }

	for(int d = 1; d<N; d++) {        // < distance between i and j
		for(int i=0; i<N; i++) {        // < starting at each i
			int j = i + d;                // < compute end j

			// stop when exceeding boundary
			if (j >= N) break;

			// find cheapest cut between i and j
			int min = RAND_MAX;
			for(int k=i; k<j; k++) {
				int costs = D[i*N+k] + D[(k+1)*N+j] + l[i] * l[k+1] * l[j+1];
				min = (costs < min) ? costs : min;
			}
			D[i*N+j] = min;
		}
	}
	double start1 = now();

		int result=D[N-1];

        // repeat for tiles (once that kernel works!)
        
        

            // clear result
            memset(C,0,sizeof(value_t) * N * N);
    
            // create buffer on device
            cl_int err;
            cl_mem devMatC = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, N * N * sizeof(int), NULL, &err);
            CLU_ERRCHECK(err, "Failed to create buffer for matrix");
            cl_mem size_list = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, S * sizeof(int), NULL, &err);
	    CLU_ERRCHECK(err, "Failed to create buffer for liste");

            // transfer data
            err = clEnqueueWriteBuffer(env.queue, size_list, CL_TRUE, 0,  S* sizeof(value_t), l, 0, NULL, NULL);
            CLU_ERRCHECK(err, "Failed to write liste to device");


            // submit kernel -> left that in as an example, need to write my own kernel now
            cl_event event;
	    const size_t global = N;            

	    cluSetKernelArguments(env.chain, 3,
                sizeof(int), &N,
                sizeof(cl_mem), (void *)&devMatC,
                sizeof(cl_mem), (void *)&size_list
            );
	    CLU_ERRCHECK(clEnqueueNDRangeKernel(env.queue, env.chain, 1, NULL, &global, NULL, 0, NULL, &event), "Failed to enqueue 2D kernel");            


            // wait for kernel
            clWaitForEvents(1,&event);
            
            // test whether kernel finished successfully
            cl_int status;
            clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
            if (status < 0) {
                CLU_ERRCHECK(-status, "Kernel failed to execute succesfully.");
                exit(1);
            }
            
            // get execution time
            cl_ulong start, end, duration;
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            duration = end - start;
   
            // release event
            CLU_ERRCHECK(clReleaseEvent(event), "Failed to release event");

            // copy results back to host -> super inefficent we just need [0][N-1] but right now not worth logic overhead
            err = clEnqueueReadBuffer(env.queue, devMatC, CL_TRUE, 0, N * N * sizeof(int), C, 0, NULL, NULL);
            CLU_ERRCHECK(err, "Failed reading back result");

            // check result
            bool success = true;

			if(result!=C[N-1]){ //primitive validation 
					success = false;
			}
                        double seconds = duration / 1e9;
            for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			if(C[N*i+j]==0){
				result=42;
			}
			printf("%i  ", C[N*i+j]);
		}
		printf("\n");
	    }


            printf("\tDuration: %2.3fs, GFLOPS: %i, Verification: %s\n", seconds, C[N-1], (success)?"OK":"FAILED");
            

            // free device memory
            CLU_ERRCHECK(clReleaseMemObject(devMatC), "Failed to release Matrix ");
            CLU_ERRCHECK(clReleaseMemObject(size_list), "Failed to release liste");

        
        
        // --- cleanup ---


        // free host memory
	free(l);
	free(C);
	free(D);

    

    // cleanup
    
    destroyMMEnvironment(env);

    // finally: report overall result
    printf("\n");
    printf("-------------------------------------------------\n");
    // done
    return EXIT_SUCCESS;
}


Matrix createMatrix(int N, int M) {
    // create data and index vector
    return malloc(sizeof(value_t)*N*M);
}

void releaseMatrix(Matrix m) {
    free(m);
}

cl_mm_environment createMMEnvironment() {

    cl_mm_environment res;
    
    // ocl initialization
    cl_device_id device_id = cluInitDeviceWithProperties(0, &res.context, &res.queue, CL_QUEUE_PROFILING_ENABLE);

    // create kernel from source
    cl_int err;
    res.program = cluBuildProgramFromFile(res.context, device_id, "chainMatOptim.cl", NULL);
    res.chain = clCreateKernel(res.program, "chainMatOptim", &err);
    CLU_ERRCHECK(err, "Failed to create mat_mul kernel from program");

    // done
    return res;
}

void destroyMMEnvironment(cl_mm_environment env) {

    // wait for completed operations (there should be none)
    CLU_ERRCHECK(clFlush(env.queue),            "Failed to flush command queue");
    CLU_ERRCHECK(clFinish(env.queue),           "Failed to wait for command queue completion");
    CLU_ERRCHECK(clReleaseKernel(env.chain),   "Failed to release kernel");
    CLU_ERRCHECK(clReleaseProgram(env.program), "Failed to release program");

    // free management resources
    CLU_ERRCHECK(clReleaseCommandQueue(env.queue), "Failed to release command queue");
    CLU_ERRCHECK(clReleaseContext(env.context),    "Failed to release OpenCL context");
}

