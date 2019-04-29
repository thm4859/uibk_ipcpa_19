#include <stdio.h>
#include <stdlib.h>

#include "cl_utils.h"
#include "utils.h"
#include "stb/image.h"
#include "stb/image_write.h"


long long roundUpToMultipleOf3(long long N, long long work_group_size, long long elements_to_check) {
    if ((N % (work_group_size*elements_to_check)) == 0) return N;
    return N + (work_group_size*elements_to_check - (N%(work_group_size*elements_to_check)));
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
    unsigned char *data = stbi_load(input_file_name, &width, &height, &components, 0);
    size_t N = width * height * components;    
    printf("Loaded image of size %dx%d with %d components.\n", width,height,components);

	//test
	//for(int i =0; i<N; i++){
	//	printf("%d = %d\n",i, data[i]);
	//}

    // start the timer
    double start_time = now();

    // ------ Analyze Image ------

    // compute min/max/avg of each component
    unsigned char min_val[components];
    unsigned char max_val[components];
    unsigned char avg_val[components];

    // an auxilary array for computing the average
    unsigned long long sum[components];

    // initialize
    for(int c = 0; c<components; c++) {
      min_val[c] = 255;
      max_val[c] = 0;
      sum[c] = 0;
    }









	//PARALLELIZE

/*	
    // compute min/max/sum
    for(int x=0; x<width; ++x) {
      for(int y=0; y<height; ++y) {
        for(int c=0; c<components; ++c) {
          unsigned char val = data[c + x*components + y*width*components];
          if (val < min_val[c]) min_val[c] = val;
          if (val > max_val[c]) max_val[c] = val;
          sum[c] += val;
        }
      }
    }
*/    
    
    

    
    
    
    
    
    // ---------- compute ----------


    printf("Counting ...\n");
	unsigned char *result;

    {
        // - setup -
        
        size_t work_group_size = 3; // components (3)
		size_t elements_to_check = 10; // min value = 3 (one work_group calculates min and max of 10 following elements of a component)

		size_t resulting_elements = 6; // 6 elements are remaining (max and min of the (3) components)		
				
		
    
		//test roundUpToMultipleOf3
		int test = N;
		int test_nach = 0;
		printf("N-start = \t\t%d\n\n", test);
		while (test > 6){
			test = roundUpToMultipleOf3(test,work_group_size,elements_to_check);
			test_nach = roundUpToMultipleOf3(test,work_group_size,elements_to_check)/(work_group_size*elements_to_check/resulting_elements);
			printf("  N = %d\tN-nach = %d\n", test, test_nach);
			test = test_nach;
		}		
		
		
		
        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue);

        // Part 2: create memory buffers
        cl_int err;
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(unsigned char), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array devDataA");
		
        cl_mem devDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, (roundUpToMultipleOf3(N,work_group_size,elements_to_check)/(work_group_size*elements_to_check/resulting_elements)) *sizeof(unsigned char), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array devDataB");
 

        // Part 3: fill memory buffers (transferring A is enough, B can be anything)
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, N * sizeof(unsigned char), data, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");

        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "auto_level.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "level", &err);
        CLU_ERRCHECK(err, "Failed to create level kernel from program");

        // Part 5: perform multi-step reduction
        clFinish(command_queue);
        timestamp begin_reduce = now();
        size_t curLength = N;
        int numStages = 0;
        while(curLength > 6) {
			
            //test
			if (curLength < 100) {
//				err = clEnqueueReadBuffer(command_queue, devDataA, CL_TRUE, 0, sizeof(unsigned char), &result, 0, NULL, NULL);
//				CLU_ERRCHECK(err, "Failed to download result from device");
//				for(int i =0; i<curLength; i++){
//					printf("%d = %d\n",i, result);
//				}
			}	
					
            // perform one stage of the reduction
            size_t global_size = roundUpToMultipleOf3(curLength,work_group_size,elements_to_check);
            
            // for debugging:
            printf("CurLength: %lu, Global: %lu, WorkGroup: %lu, elements_to_check: %lu\n", curLength, global_size, work_group_size, elements_to_check);
        
            // update kernel parameters
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &devDataA);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &devDataB);
            clSetKernelArg(kernel, 2, work_group_size * elements_to_check * sizeof(cl_mem), NULL);
            clSetKernelArg(kernel, 3, sizeof(size_t), &curLength);
            clSetKernelArg(kernel, 4, sizeof(size_t), &work_group_size);
            clSetKernelArg(kernel, 5, sizeof(size_t), &elements_to_check);
        
            // submit kernel
            CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");
        
            // update curLength
            curLength = global_size / (work_group_size*elements_to_check/resulting_elements);
            
            // swap buffers
            cl_mem tmp = devDataA;
            devDataA = devDataB;
            devDataB = tmp;
            
            // count number of steps
            numStages++;
            

			
        }
        clFinish(command_queue);
        timestamp end_reduce = now();
        printf("\t%d stages of reductions took: %.3f ms\n", numStages, (end_reduce-begin_reduce)*1000);

        
        // download result from device
        err = clEnqueueReadBuffer(command_queue, devDataA, CL_TRUE, 0, sizeof(unsigned char) * 6, &result, 0, NULL, NULL);
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
    
    //write ocl output to array
    for(int c = 0; c<components; c++) {
      min_val[c] = result[c];
      max_val[c] = result[c + components];
    }    
      sum[0] = 126;
      sum[1] = 147;
      sum[2] = 153; 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    // compute average and multiplicative factors
    float min_fac[components];
    float max_fac[components];
    for(int c=0; c<components; ++c) {
      avg_val[c] = sum[c]/((unsigned long long)width*height);
      min_fac[c] = (float)avg_val[c] / (float)(avg_val[c] - min_val[c]);
      max_fac[c] = (255.0f-(float)avg_val[c]) / (float)(max_val[c] - avg_val[c]);
      printf("\tComponent %1u: %3u / %3u / %3u * %3.2f / %3.2f\n", c, min_val[c], avg_val[c], max_val[c], min_fac[c], max_fac[c]);
    }

    // ------ Adjust Image ------

    for(int x=0; x<width; ++x) {
      for(int y=0; y<height; ++y) {
        for(int c=0; c<components; ++c) {
          int index = c + x*components + y*width*components;
          unsigned char val = data[index];
          float v = (float)(val - avg_val[c]);
          v *= (val < avg_val[c]) ? min_fac[c] : max_fac[c];
          data[index] = (unsigned char)(v + avg_val[c]);
        }
      }
    }

    printf("Done, took %.1f ms\n", (now() - start_time)*1000.0);

    // ------ Store Image ------

    printf("Writing output image %s ...\n", output_file_name);
    stbi_write_png(output_file_name,width,height,components,data,width*components);
    stbi_image_free(data);

    printf("Done!\n");

    // done
    return EXIT_SUCCESS;
}
