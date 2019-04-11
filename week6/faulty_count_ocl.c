#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    long long N = 10;//actual problemsize
    long long M = 2;//convenient roundup
    int groups=8;
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
    
    while(M<N){
    	M=M*2;
    }
    load = M/groups;
    
    printf("M: %lld \n",M);
    // ---------- setup ----------

    // create two input vectors (on heap!)
    int* res = malloc(sizeof(int)*groups); //just to have an overhead full of 0s 
    int* a = malloc(sizeof(int)*M);
    
    // fill array
    for(long long i = 0; i<N; i++) {
        a[i] = rand() % 2;
    }
    

    // ---------- compute ----------
    
   
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
        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);



        // Part B - data management
        
        // 5) create memory buffers on device
        size_t vec_size = sizeof(int) * M;
        cl_mem devVecA = clCreateBuffer(context, CL_MEM_READ_WRITE , vec_size, NULL, &ret);
        cl_mem devVecRet = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(int)*groups, NULL, &ret);

        // 6) transfer input data from host to device (synchronously)
        ret = clEnqueueWriteBuffer(command_queue, devVecA, CL_TRUE, 0, vec_size, &a[0], 0, NULL, NULL);



        // Part C - computation

        // 6) load kernel code from file
        kernel_code code = loadCode("count.cl");
        
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
        kernel = clCreateKernel(program, "count", &ret);
	//ok one call reduces it to #groups -> if that is < load than 1 last call to get 1 number back
	// otherwise take group as new input, and leave load the same

        // 10) set arguments
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &devVecA);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devVecRet);
        ret = clSetKernelArg(kernel, 2, sizeof(int)*load, NULL);


        // 11) schedule kernel
        size_t global_work_offset = 0;
        size_t global_work_size = M;
        size_t local_work_size = load;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 
                    1, &global_work_offset, &global_work_size, &local_work_size, 
                    0, NULL, NULL
        );
	 ret = clEnqueueReadBuffer(command_queue, devVecRet, CL_TRUE, 0, sizeof(int)*groups, &res[0], 0, NULL, NULL);
	for(int i=0;i<groups;i++){
		printf("%i\n",res[i]);
	}
	printf("%i_______________\n",groups);
	//some other error occurs around 1 000 000 000
	//so have it right now hardcoded at 3 flips
		 global_work_offset = 0;
		 global_work_size = groups;
		 local_work_size = load;
		if(groups<load){
			groups=1;
			local_work_size = 1;
		}else{
			groups=groups/load;//now its the new returnsize
			if(groups==0){
			    groups=1;
			    local_work_size = 1;	
			}
		}
		
		cl_mem devVecB = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(int)*groups, NULL, &ret);//2nd round return buffer
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &devVecRet);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devVecB);

		ret = clEnqueueNDRangeKernel(command_queue, kernel, //so worst case this has reduces per factor 1 000 000 and has still 1 stage to go
		            1, &global_work_offset, &global_work_size, &local_work_size, 
		            0, NULL, NULL
		);
			 ret = clEnqueueReadBuffer(command_queue, devVecB, CL_TRUE, 0, sizeof(int)*groups, &res[0], 0, NULL, NULL);
		for(int i=0;i<groups;i++){
			printf("%i\n",res[i]);
		}
		printf("%i_______________\n",groups);
		if(groups!=1){ //checking the case that 1st reduction was already enough and second step just reduced by #group -> no third stage neccesary
		//obviously somewhere here in either worksizes or bufferflags there must be an error because second readbuffer returns nothing (in print thats the old result)
			global_work_offset = 0;
		 	global_work_size = groups;
		 	local_work_size = load;
			if(groups<load){
				groups=1;
				local_work_size = 1;
			}else{
				groups=groups/load;
				if(groups==0){
				    groups=1;
				    local_work_size = 1;
				}	
			}
		
		
			cl_mem devVecC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*groups, NULL, &ret);
			ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &devVecB);
			ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devVecC);

			ret = clEnqueueNDRangeKernel(command_queue, kernel, 
		            1, &global_work_offset, &global_work_size, &local_work_size, 
		            0, NULL, NULL
			);
				 ret = clEnqueueReadBuffer(command_queue, devVecC, CL_TRUE, 0, sizeof(int)*groups, &res[0], 0, NULL, NULL);
			for(int i=0;i<groups;i++){
				printf("%i\n",res[i]);
			}
			printf("%i_______________\n",ret); //so no errorcode yet only the first element is used in second stages
        		ret = clEnqueueReadBuffer(command_queue, devVecC, CL_TRUE, 0, sizeof(int)*groups, &res[0], 0, NULL, NULL);
			ret = clReleaseMemObject(devVecB);
			ret = clReleaseMemObject(devVecC);

		}else{
			
		        ret = clEnqueueReadBuffer(command_queue, devVecB, CL_TRUE, 0, sizeof(int)*groups, &res[0], 0, NULL, NULL);
			ret = clReleaseMemObject(devVecB);
		}
		ret = clReleaseMemObject(devVecRet);
	

        // 12) transfer data back to host
	ret = clFlush(command_queue);
	
        // Part D - cleanup
        
        // wait for completed operations (should all have finished already)
        
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        
        // free device memory
        ret = clReleaseMemObject(devVecA);
        
        
        // free management resources
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

    }
    timestamp end = now();
    printf("Total time: %.3f ms\n", (end-begin)*1000);

    // ---------- check ----------    
    
    bool success = true;
    long long cnt = 0;
    for (int i = 0; i < N; i++) {
        int entry = a[i];
        if (entry == 1)
            cnt++;
    }

    if(res[0]==cnt){
	printf("validation OK \n");
	success=true;
    }else{
	success=false;
	printf("validation FALSE\n");
    }
    printf("Result: %i vs %lld %i\n",res[0],cnt,groups);
    

    
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
