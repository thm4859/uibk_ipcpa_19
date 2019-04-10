#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "utils.h"
#include "cl_utils.h"

typedef float value_t;

// -- kernel code utils --

typedef struct kernel_code {
    const char* code;
    size_t size;
} kernel_code;

kernel_code loadCode(const char* filename);

void releaseCode(kernel_code code);
unsigned long long getElapsed(cl_event event);

// -----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    long long N = 100*1000*1000;//actual problemsize
    long long M = 2;//convenient roundup
    int groups=8;
    cl_event event;
    
    if (argc > 1) {
        N = atoll(argv[1]);
    }
    if(argc >2){
	groups=atoi(argv[2]);
    }
    srand((unsigned) time(NULL));
    if(argc >3){
	srand(atoi(argv[3]));//give a random seed -> to replicate result
    }
    printf("Computing vector-add with N=%lld\n", N);
    
    int load = N/groups;
    while(load>(1024)){ //this factors a lot of unknows -> localmemory on hardware, size of float and so on...
	groups=groups*2;
	load = N/groups;
    }    
    printf("With %d workgroups\n", groups);
    if(N!=groups*load){
	while(M<N){
	    	M=M*2;
	}
	load = M/groups;
    }else{
	M=N;
    }
    printf("M: %lld \n",M);
    // ---------- setup ----------

    // create two input vectors (on heap!)
    int* res = malloc(sizeof(int)*groups); //just to have an overhead full of 0s 
    value_t* a = malloc(sizeof(value_t)*M);
    
    // fill array
    for(long long i = 0; i<N; i++) {
        a[i] = rand() % 2;
    }
    

    // ---------- compute ----------
    unsigned long long event_run_kernel = 0.0f;
   
    // --- OpenCL part ---
    timestamp begin = now();
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
        //command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
        command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
		CLU_ERRCHECK(ret, "Failed to enable CL_QUEUE_PROFILING_ENABLE on command queue");



        // Part B - data management
        
        // 5) create memory buffers on device
        size_t vec_size = sizeof(value_t) * M;
        cl_mem devVecA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, vec_size, NULL, &ret);
        CLU_ERRCHECK(ret, "Failed to create buffer for devVecA");
        cl_mem devVecRet = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(value_t)*groups, NULL, &ret);
		CLU_ERRCHECK(ret, "Failed to create buffer for devVecRet");

        // 6) transfer input data from host to device (synchronously)
        ret = clEnqueueWriteBuffer(command_queue, devVecA, CL_TRUE, 0, vec_size, &a[0], 0, NULL, NULL);
		CLU_ERRCHECK(ret, "Failed to write matrix A to device");


        // Part C - computation

        // 6) load kernel code from file
        kernel_code code = loadCode("count.cl");
        
        // 7) compile kernel program from source
        program = clCreateProgramWithSource(context, 1, &code.code,
				                      (const size_t *)&code.size, &ret);
		CLU_ERRCHECK(ret, "Failed to clCreateProgramWithSource()");

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
        kernel = clCreateKernel(program, "count", &ret);
        CLU_ERRCHECK(ret, "Failed to create COUNT kernel from program");
	
        // 10) set arguments
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &devVecA);
        CLU_ERRCHECK(ret, "Failed to clSetKernelArg 0");
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devVecRet);
        CLU_ERRCHECK(ret, "Failed to clSetKernelArg 1");
        ret = clSetKernelArg(kernel, 2, sizeof(int)*load, NULL);
        CLU_ERRCHECK(ret, "Failed to clSetKernelArg 2");
        //ret = clSetKernelArg(kernel, 3, sizeof(int), &M);
        //CLU_ERRCHECK(ret, "Failed to clSetKernelArg 3");

        // 11) schedule kernel
        size_t global_work_offset = 0;
        size_t global_work_size = M;
        size_t local_work_size = load;
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 
                    1, &global_work_offset, &global_work_size, &local_work_size, 
                    0, NULL, &event
        ), "Failed to enqueue 2D kernel");
	
        // 12) transfer data back to host
		ret = clFlush(command_queue);
		CLU_ERRCHECK(ret, "Failed to clFlush(command_queue)");
        ret = clEnqueueReadBuffer(command_queue, devVecRet, CL_TRUE, 0, sizeof(int)*groups, &res[0], 0, NULL, NULL);
        CLU_ERRCHECK(ret, "Failed to clEnqueueReadBuffer()");
        
        // Part D - cleanup
        
        // wait for completed operations (should all have finished already)
        
        ret = clFinish(command_queue);
        CLU_ERRCHECK(ret, "Failed to flush command queue");
        ret = clReleaseKernel(kernel);
        CLU_ERRCHECK(ret, "Failed to release kernel");
        ret = clReleaseProgram(program);
        CLU_ERRCHECK(ret, "Failed to release program");
        
        // free device memory
        ret = clReleaseMemObject(devVecA);
        CLU_ERRCHECK(ret, "Failed to release devVecA");
        ret = clReleaseMemObject(devVecRet);
        CLU_ERRCHECK(ret, "Failed to release devVecRet");
        
        // free management resources
        ret = clReleaseCommandQueue(command_queue);
        CLU_ERRCHECK(ret, "Failed to release command queue");
        ret = clReleaseContext(context);
        CLU_ERRCHECK(ret, "Failed to release OpenCL context");

    }
    timestamp end = now();
    printf("Total time: %.3f ms\n", (end-begin)*1000);

    // ---------- check ----------    
    
    //kernel runtime
    event_run_kernel = getElapsed(event);
    printf("Kernel time: %.3f ms\n", (event_run_kernel/1e6));
    
    bool success = true;
    int result=0;
    long long cnt = 0;
    for (int i = 0; i < N; i++) {
        int entry = a[i];
        if (entry == 1)
            cnt++;
    }
    for(int i = 0; i<groups; i++) {
	result+=res[i];
    }
    if(result==cnt){
	printf("validation OK \n");
	success=true;
    }else{
	success=false;
	printf("validation FALSE\n");
    }
    printf("Result: %i \n",result);
    

    
    // ---------- cleanup ----------
    
    free(a);
    free(res);
    
    // done
    return (success) ? EXIT_SUCCESS : EXIT_FAILURE;
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

unsigned long long getElapsed(cl_event event) {
    cl_ulong starttime = 0, endtime = 0;
    CLU_ERRCHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL), "Failed to get profiling information");
    CLU_ERRCHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL), "Failed to get profiling information");
	return (endtime-(unsigned long long)starttime);
}
