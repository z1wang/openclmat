////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#include <stdbool.h>

////////////////////////////////////////////////////////////////////////////////
#define WA 5  /*common dimension*/
#define HA 5 /*A's row*/
#define WB 5 /*B's column*/

#define HB WA
#define WC WB
#define HC HA
////////////////////////////////////////////////////////////////////////////////

/**/
void ttoo(float** a, float* b, int h, int w){
  int i, j;
  for(i = 0; i < h; i++){
    for(j = 0; j < w; j++){
      b[i*w + j] = a[i][j];
    }
  }
}

void otot(float* a, float** b, int h, int w){
  int i, j;
  for(i = 0; i < h; i++){
    for(j = 0; j < w; j++){
      b[i][j] = a[i*w + j];
    }
  }
}

/*  x is of n*m;
  w is of (m) * k
*/
void forward1(float **z2, float **x, float **w, int n, int m, int k){
  int i, j, a;
  for(i = 0; i < n; i++){
    for(j = 0; j < k; j++){
      for(a = 0; a < m; a++){
        z2[i][j] += x[i][a] * w[a][j];
      }
    }
  }
  return;
}

// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
   int i;

   for (i = 0; i < size; ++i)
   	data[i] = rand() / (float)RAND_MAX;
}

long LoadOpenCLKernel(char const* path, char **buf, bool add_nul)
{
    FILE  *fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fp = fopen(path, "r");
    if( NULL == fp ) {
        return -1L;
    }

    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }

    /* Byte offset to the end of the file (size) */
    if( 0 > (off_end = ftell(fp)) ) {
        return -1L;
    }
    fsz = (size_t)off_end;

    /* Allocate a buffer to hold the whole file */
    *buf = (char *) malloc( fsz+(int)add_nul );
    if( NULL == *buf ) {
        return -1L;
    }

    /* Rewind file pointer to start of file */
    rewind(fp);

    /* Slurp file into buffer */
    if( fsz != fread(*buf, 1, fsz, fp) ) {
        free(*buf);
        return -1L;
    }

    /* Close the file */
    if( EOF == fclose(fp) ) {
        free(*buf);
        return -1L;
    }

    if( add_nul ) {
        /* Make sure the buffer is NUL-terminated, just in case */
        *buf[fsz] = '\0';
    }

    /* Return the file size */
    return (long)fsz;
}

int main(int argc, char** argv)
{
   int err;                            // error code returned from api calls

   cl_device_id device_id;             // compute device id 
   cl_context context;                 // compute context
   cl_command_queue commands;          // compute command queue
   cl_program program;                 // compute program
   cl_kernel kernel;                   // compute kernel

    // OpenCL device memory for matrices
   cl_mem d_A;
   cl_mem d_B;
   cl_mem d_C;

   // set seed for rand()
   srand(2014);
 
   //Allocate host memory for matrices A and B
   unsigned int size_A = WA * HA;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = WB * HB;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);

   //Initialize host memory
   randomMemInit(h_A, size_A);
   randomMemInit(h_B, size_B);
 
   //Allocate host memory for the result C
   unsigned int size_C = WC * HC;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C = (float*) malloc(mem_size_C);

   float* h_D = (float*) malloc(mem_size_C);
   float** d_h_D = (float**) malloc(HC * sizeof(float*));
   int ccc;
   for(ccc = 0; ccc < HC; ccc++){
    d_h_D[ccc] = (float*) malloc(WC * sizeof(float));
   }

   float** d_h_A = (float**) malloc(HA * sizeof(float*));
   for(ccc = 0; ccc < HA; ccc++){
    d_h_A[ccc] = (float*) malloc(WA * sizeof(float));
   }

   float** d_h_B = (float**) malloc(HB * sizeof(float*));
   for(ccc = 0; ccc < HB; ccc++){
    d_h_B[ccc] = (float*) malloc(WB * sizeof(float));
   }

  
   printf("Initializing OpenCL device...\n"); 

   cl_uint dev_cnt = 0;
   clGetPlatformIDs(0, 0, &dev_cnt);
	
   cl_platform_id platform_ids[100];
   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	
   // Connect to a compute device
   int gpu = 1;
   err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to create a device group!\n");
       return EXIT_FAILURE;
   }
  
   // Create a compute context 
   context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
   if (!context)
   {
       printf("Error: Failed to create a compute context!\n");
       return EXIT_FAILURE;
   }

   // Create a command commands
   commands = clCreateCommandQueue(context, device_id, 0, &err);
   if (!commands)
   {
       printf("Error: Failed to create a command commands!\n");
       return EXIT_FAILURE;
   }

   // Create the compute program from the source file
   char *KernelSource;
   long lFileSize;

   lFileSize = LoadOpenCLKernel("matrixmul_kernel.cl", &KernelSource, false);
   if( lFileSize < 0L ) {
       perror("File read failed");
       return 1;
   }

   program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
   if (!program)
   {
       printf("Error: Failed to create compute program!\n");
       return EXIT_FAILURE;
   }

   // Build the program executable
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   printf("%d\n", err);
   if (err != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];
       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       exit(1);
   }

   // Create the compute kernel in the program we wish to run
   //
   kernel = clCreateKernel(program, "matrixMul", &err);
   if (!kernel || err != CL_SUCCESS)
   {
       printf("Error: Failed to create compute kernel!\n");
       exit(1);
   }

   // Create the input and output arrays in device memory for our calculation
   d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
   d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &err);
   d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &err);

   if (!d_A || !d_B || !d_C)
   {
       printf("Error: Failed to allocate device memory!\n");
       exit(1);
   }    
    
   printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", WA,HA,WB,HB); 

   //Launch OpenCL kernel
   size_t localWorkSize[2], globalWorkSize[2];
 
   int wA = WA;
   int wC = WC;
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&wA);
   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&wC);
//   err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&wC);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to set kernel arguments! %d\n", err);
       exit(1);
   }
 
   localWorkSize[0] = 1;
   localWorkSize[1] = 1;
   globalWorkSize[0] = HC;
   globalWorkSize[1] = WC;
 
   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to execute kernel! %d\n", err);
       exit(1);
   }
 
   //Retrieve result from device
   err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to read output array! %d\n", err);
       exit(1);
   }


    int i;
    otot(h_A, d_h_A, HA, WA);
    otot(h_B, d_h_B, HB, WB);

    forward1(d_h_D, d_h_A, d_h_B, HA, WA, WB);
    ttoo(d_h_D, h_D, HC, WC);

 
   //print out the results
   printf("\n\nMatrix A\n");

   for(i = 0; i < size_A; i++)
   {
      printf("%f ", h_A[i]);
      if(((i + 1) % WA) == 0)
      printf("\n");
   }
   printf("\n");

  printf("\n\nMatrix B\n");
   for(i = 0; i < size_B; i++)
   {
      printf("%f ", h_B[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
   }
   printf("\n");


   printf("\n\nMatrix C (Results)\n");
   for(i = 0; i < size_C; i++)
   {
      printf("%f ", h_C[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
   }
   printf("\n");

 /*  printf("\n\nMatrix D (Results)\n");
   for(i = 0; i < size_C; i++)
   {
      printf("%f ", h_D[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
   }
   printf("\n");*/

  
   printf("Matrix multiplication completed...\n"); 

   //Shutdown and cleanup
   free(h_A);
   free(h_B);
   free(h_C);
 
   clReleaseMemObject(d_A);
   clReleaseMemObject(d_C);
   clReleaseMemObject(d_B);

   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(commands);
   clReleaseContext(context);

   return 0;
}
