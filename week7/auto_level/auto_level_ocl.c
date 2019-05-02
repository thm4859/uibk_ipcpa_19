#include <stdio.h>
#include <stdlib.h>

#include "cl_utils.h"
#include "utils.h"
#include "stb/image.h"
#include "stb/image_write.h"

long long roundUpToMultiple(long long N, long long B) {
    if ((N % B) == 0) return N;
    return N + (B - (N%B));
}

long long roundUpToMultipleOfx(long long N, long long components, long long elements_to_check) {
    if ((N % (components*elements_to_check)) == 0) return N;
    return N + (components*elements_to_check - (N%(components*elements_to_check)));
}

int main(int argc, char** argv) {

    // parse input parameters
    if(argc != 3) {
      printf("Usage: %s input.png output.png\n", argv[0]);
      return EXIT_FAILURE;
    }

    char* input_file_name = argv[1];
    char* output_file_name = argv[2];


    // load input file
    printf("Loading input file %s ..\n", input_file_name);
    int width, height, components;
    unsigned char *data_uchar = stbi_load(input_file_name, &width, &height, &components, 0);
    float time_min_max = 0.0f, time_sum = 0.0f, time_adjust = 0.00f;
    size_t N = width * height * components; 
    size_t comp = (size_t)components; 
    
    unsigned long *data = (unsigned long*)malloc(N*sizeof(unsigned long));			//used for: "max and min" and "sum" to fill devDataA (-> filled im line 44+)
    float *data_float = (float*)malloc(N*sizeof(float));
    
    float *min_fac = (float*)malloc(components*sizeof(float));
    float *max_fac = (float*)malloc(components*sizeof(float));
    for (int i = 0; i < N; i++) {
<<<<<<< HEAD
		data[i] = (unsigned long)data_uchar[i];					//used for: "max and min" and "sum" to fill devDataA
		data_float [i] = (float)data_uchar[i];					//used for: "adjust" to fill devDataA
		
=======
		data[i] = (unsigned long)data_uchar[i];
		data_float [i] = (float)data_uchar[i];
>>>>>>> 21123ffbe1d3364dad48db6f38fb4a2724da7d7c
	}
	
    
    
    printf("Loaded image of size %dx%d with %d components.\n", width,height,components);

	//printf("erstes element: %lu\n", data[0]);
	
	
    // start the timer
    double start_time = now();

    // ------ Analyze Image ------

    // compute min/max/avg of each component
    unsigned char min_val[components];
    unsigned char max_val[components];
    unsigned char avg_val[components];
    float avg_val_float [components];
    for (int i = 0; i < components; i++) {
		avg_val_float [i] = (float)avg_val[i];
	}

    // an auxilary array for computing the average
    unsigned long long sum[components];
    size_t work_group_size = 30; 
	size_t elements_to_check = 10; // min value = 3 (one work_group calculates min and max of 10 following elements of a component)
	float *data_res_float = (float*)malloc(roundUpToMultipleOfx(N,components,elements_to_check)*sizeof(float)); 		//not used at the moment
    // initialize
    for(int c = 0; c<components; c++) {
      min_val[c] = 255;
      max_val[c] = 0;
      sum[c] = 0;
    }



    // ---------- compute max and min of .png data array----------

    printf("\n\n...Computing min max of png-data ...\n");
	unsigned long min_max[components * 2];
	
    {
        // - setup -
		size_t resulting_elements = 6; // 6 elements are remaining (max and min of the (3) components)		
				
		//test roundUpToMultipleOfx
		//int test = N;
		//int N_test = roundUpToMultipleOfx(test,components,elements_to_check)/(components*elements_to_check/resulting_elements);
		//int test_nach = 0;
		//printf("N-start = \t\t%d\n\n", test);
		
		//print the the dimension N of the data array after the reduction steps
		//while (test > resulting_elements){
		//	test = roundUpToMultipleOfx(test,components,elements_to_check);
		//	test_nach = roundUpToMultipleOfx(test,components,elements_to_check)/(components*elements_to_check/resulting_elements);
		//	printf("  N = %d\tN-nach = %d\n", test, test_nach);
		//	test = test_nach;
		//}		
		
		
		
        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue);

        // Part 2: create memory buffers
        cl_int err;
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(unsigned long), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array devDataA");
		
        cl_mem devDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, (roundUpToMultipleOfx(N,components,elements_to_check)/(components*elements_to_check/resulting_elements)) * sizeof(unsigned long), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array devDataB");
 

        // Part 3: fill memory buffers (transferring A is enough, B can be anything)
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, N * sizeof(unsigned long), data, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");

        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "auto_level.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "level", &err);
        CLU_ERRCHECK(err, "Failed to create level kernel from program");

        // Part 5: perform multi-step reduction
        clFinish(command_queue);
        timestamp begin_min_max = now();
        size_t curLength = N;
        int numStages = 0;
        while(curLength > resulting_elements) {
			
            // perform one stage of the reduction
            size_t global_size = roundUpToMultipleOfx(curLength,components,elements_to_check);
            
            // for debugging:
            //printf("CurLength: %lu, Global: %lu, WorkGroup: %lu, elements_to_check: %lu\n", curLength, global_size, work_group_size, elements_to_check);
        
            // update kernel parameters
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &devDataA);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &devDataB);
            clSetKernelArg(kernel, 2, components * elements_to_check * sizeof(unsigned long), NULL);
            clSetKernelArg(kernel, 3, sizeof(size_t), &curLength);
            clSetKernelArg(kernel, 4, sizeof(size_t), &components);
            clSetKernelArg(kernel, 5, sizeof(size_t), &elements_to_check);
        
            // submit kernel
            CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");
        
            // update curLength
            curLength = global_size / (components*elements_to_check/resulting_elements);
            
            // swap buffers
            cl_mem tmp = devDataA;
            devDataA = devDataB;
            devDataB = tmp;
            
            // count number of steps
            numStages++;
            
            //print of last reduction step data
			//if(curLength == 6) {
			//	printf("CurLength: %lu, Global: %lu, WorkGroup: %lu, elements_to_check: %lu\n", resulting_elements, global_size, work_group_size, elements_to_check);
			//}
			
        }        
        clFinish(command_queue);
        timestamp end_min_max = now();
        time_min_max = (end_min_max-begin_min_max)*1000;
        printf("\tMIN MAX: %d stages of reductions took: %.3f ms\n", numStages, (end_min_max-begin_min_max)*1000);
        


        
        // download result from device
        err = clEnqueueReadBuffer(command_queue, devDataA, CL_TRUE, 0, resulting_elements * sizeof(cl_ulong), &min_max, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to download result from device");


        // -------- print result -------
		min_val[0] = (unsigned char)min_max[0];
		min_val[1] = (unsigned char)min_max[1];
		min_val[2] = (unsigned char)min_max[2];
		max_val[0] = (unsigned char)min_max[3];
		max_val[1] = (unsigned char)min_max[4];
		max_val[2] = (unsigned char)min_max[5];
		
		//for (int i = 0; i < components; i++) {		
		//	printf("\tComponent %d: %lu / %lu\n", i, min_val[i], max_val[i]);
		//}


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




    
    
    // ---------- compute sum of png-data----------
    {
		printf("\n\n...Computing sum of png-data ...\n");
		unsigned long count_sum[components];
		timestamp begin_sum = now();	
		
        // - setup -
        
		size_t resulting_elements = 3; // 6 elements are remaining (max and min of the (3) components)	

        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue); 

        // Part 2: create memory buffers
        cl_int err;
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(unsigned long), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");

        cl_mem devDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, roundUpToMultipleOfx(N,components,elements_to_check) *sizeof(unsigned long), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");


        // Part 3: fill memory buffers (transferring A is enough, B can be anything)
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, N * sizeof(unsigned long), data, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");

        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "auto_level_sum.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "sum", &err);
        CLU_ERRCHECK(err, "Failed to create reduction kernel from program");

        // Part 5: perform multi-step reduction
        clFinish(command_queue);
        timestamp begin_reduce = now();
        size_t curLength = N;
        int numStages = 0;
        while(curLength > components) {

            // perform one stage of the reduction
            size_t global_size = roundUpToMultipleOfx(curLength,components,elements_to_check);
            
            // for debugging:
            //printf("CurLength: %lu, Global: %lu, WorkGroup: %lu, elements_to_check: %lu\n", curLength, global_size, work_group_size, elements_to_check);
        
            // update kernel parameters
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &devDataA);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &devDataB);
            clSetKernelArg(kernel, 2, components * elements_to_check * sizeof(unsigned long), NULL);
            clSetKernelArg(kernel, 3, sizeof(size_t), &curLength);
            clSetKernelArg(kernel, 4, sizeof(size_t), &components);
            clSetKernelArg(kernel, 5, sizeof(size_t), &elements_to_check);
        
            // submit kernel
            CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");
        
            // update curLength
            curLength = global_size / (components*elements_to_check/resulting_elements);
            
            // swap buffers
            cl_mem tmp = devDataA;
            devDataA = devDataB;
            devDataB = tmp;
            
            // count number of steps
            numStages++;
            
            //print of last reduction step data
			//if(curLength == 3) {
			//	printf("CurLength: %lu, Global: %lu, WorkGroup: %lu, elements_to_check: %lu\n", resulting_elements, global_size, work_group_size, elements_to_check);
			//}
			
			//for error testing
			//err = clFinish(command_queue);
			//printf("ERRORCODE = %d\n", err);
			
        }

        timestamp end_sum = now();
        time_sum = (end_sum-begin_sum)*1000;
        printf("\tSUM: %d stages of reductions took: %.3f ms\n", numStages, (end_sum-begin_sum)*1000);

        
        // download result from device
        err = clEnqueueReadBuffer(command_queue, devDataA, CL_TRUE, 0, components * sizeof(cl_ulong), &count_sum, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to download result from device");

		// -------- print result -------
		for (int i = 0; i < components; i++) {
			sum[i] = (unsigned long long)count_sum[i];
			printf("\tsum of component %d: %lu\t(mean value: %lu)\n", i, count_sum[i], count_sum[i]/(width * height));
		}
			  
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


    
    // compute average and multiplicative factors
    //float min_fac[components];
    //float max_fac[components];
    for(int c=0; c<components; ++c) {
		avg_val_float[c] = (float)sum[c]/((unsigned long long)width*height);
		avg_val[c] = sum[c]/((unsigned long long)width*height);
		min_fac[c] = (float)avg_val[c] / (float)(avg_val[c] - min_val[c]);
		max_fac[c] = (255.0f-(float)avg_val[c]) / (float)(max_val[c] - avg_val[c]);
		printf("\tComponent %d: %3u / %3u / %3u * %3.2f / %3.2f\n", c, min_val[c], avg_val[c], max_val[c], min_fac[c], max_fac[c]);
    }
    









    // ---------- compute adjust of .png data array----------

    printf("\n\n...Computing adjust of png-data ...\n");

    {
		
        // - setup -
		//size_t global_size = N;	
		size_t global_size = roundUpToMultipleOfx(N,components,elements_to_check);	
		
        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue);

        // Part 2: create memory buffers
        cl_int err;
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_WRITE, global_size * sizeof(unsigned long), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array devDataA");
		
        cl_mem devDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(unsigned char), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array devDataB");
 
        cl_mem dev_min_fac = clCreateBuffer(context, CL_MEM_READ_ONLY, components * sizeof(float), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array dev_min_fac");
        
        cl_mem dev_max_fac = clCreateBuffer(context, CL_MEM_READ_ONLY, components * sizeof(float), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array dev_max_fac");
        
        cl_mem dev_avg_val = clCreateBuffer(context, CL_MEM_READ_ONLY, components * sizeof(float), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array ddef_avg_val"); 


        // Part 3: fill memory buffers (transferring A is enough, B can be anything)
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, global_size * sizeof(float), data_float, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");
        err = clEnqueueWriteBuffer(command_queue, dev_min_fac, CL_TRUE, 0, components * sizeof(float), min_fac, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");
        err = clEnqueueWriteBuffer(command_queue, dev_max_fac, CL_TRUE, 0, components * sizeof(float), max_fac, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");
        err = clEnqueueWriteBuffer(command_queue, dev_avg_val, CL_TRUE, 0, components * sizeof(float), avg_val_float, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");

        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "auto_level_adjust.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "adjust", &err);
        CLU_ERRCHECK(err, "Failed to create level kernel from program");

        // Part 5: perform multi-step reduction
        clFinish(command_queue);
        timestamp begin_adjust = now();
            
        // for debugging:
        printf("Data-dimension: %ld, Global: %lu, WorkGroup: %lu, elements_to_check: %lu\n", N, global_size, work_group_size, elements_to_check);
        
        // update kernel parameters
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &devDataA);
        CLU_ERRCHECK(err, "Failed to write clSetKernelArg 0");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devDataB);
		CLU_ERRCHECK(err, "Failed to write clSetKernelArg 1");
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_min_fac);
        CLU_ERRCHECK(err, "Failed to write clSetKernelArg 2");
        err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dev_max_fac);
        CLU_ERRCHECK(err, "Failed to write clSetKernelArg 3");
        err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &dev_avg_val);  
        CLU_ERRCHECK(err, "Failed to write clSetKernelArg 4");      
        err = clSetKernelArg(kernel, 5, components * elements_to_check * sizeof(float), NULL);
        CLU_ERRCHECK(err, "Failed to write clSetKernelArg 5");
        err = clSetKernelArg(kernel, 6, sizeof(size_t), &components);
        CLU_ERRCHECK(err, "Failed to write clSetKernelArg 6");
        err = clSetKernelArg(kernel, 7, sizeof(size_t), &N);


                
        // submit kernel
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue adjust kernel");
        
                
        //for error testing
		err = clFinish(command_queue);
		//printf("ERRORCODE = %d\n", err);
        
        //clFinish(command_queue);
        timestamp end_adjust = now();
        time_min_max = (end_adjust-begin_adjust)*1000;
        printf("\tADJUST: took: %.3f ms\n", (end_adjust-begin_adjust)*1000);
        printf("\t---ocl-time: %.3f ms---\n", time_min_max + time_sum + time_adjust);
        
        
        // download result from device
        err = clEnqueueReadBuffer(command_queue, devDataB, CL_TRUE, 0, N * sizeof(cl_uchar), data_uchar, 0, NULL, NULL);
        //printf("ERRORCODE = %d\n", err);
        CLU_ERRCHECK(err, "Failed to download result from device");
		
		
		//for (int i = 0; i < N; i+100000) {
		//	printf("res: %f3.0\n", data_res_float[i]);
		//}


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



	//printf("ocl-time: %f ms\n", time_min_max + time_sum + time_adjust);

    // ------ Store Image ------

    printf("Writing output image %s ...\n", output_file_name);
    stbi_write_png(output_file_name,width,height,components,data_uchar,width*components);
    stbi_image_free(data_uchar);
    free(data);
    free(data_float);
   
    printf("Done!\n");

    // done
    return EXIT_SUCCESS;
}
