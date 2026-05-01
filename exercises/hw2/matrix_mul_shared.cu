#include <stdio.h>

// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 8192;
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block
const float A_val = 3.0f;
const float B_val = 2.0f;

// matrix multiply kernel: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {
  // - The input matrices A and B have been flattened into row-major arrays
  //   (i.e. the values of row 0, followed by the values of row 1, etc.).
  //   The output matrix C will be written in the same format.
  // - ds is the dimension of the matrices (all are square)

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  // Each thread will compute a single output element
  // We can use the x index to define the output element's column and y for its row
  // (this is convention)
  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  // The approach from week 1 involved repeatedly reading the same elements from
  // global memory. The below approach uses shared memory to reuse the same data
  // across the threads within a block. Specifically, we load a small square tile
  // of A and B into shared memory, use it fully, then move on to the next tile.  
  if ((idx < ds) && (idy < ds)){
    float temp = 0;

    // Loop over all of the tiles (assumes ds is perfectly divisible by tile size)
    // Each tile has dimension block_size*block_size (i.e. one element for every
    // thread in the block).
    for (int i = 0; i < ds/block_size; i++) {

      // Every iteration of this loop loads a new 32x32 tile of A and 32x32 tile of B.
      // (remember threadIdx.x and .y range from 0 to block_size-1)
      //
      // Each iteration, the A tile moves RIGHT, whilst the B tile moves DOWN
      // For A:
      //    - idy*ds:           goes to the row this thread is responsible for
      //    - (i * block_size): moves horizontal to the start of the current tile
      //    - threadIdx.x:      moves horizontal to the local column this thread is responsible for
      // For B:
      //    - (i * block_size)*ds:  moves down to the start of the current tile
      //    - threadIdx.y*ds:       goes to the local row this thread is responsible for
      //    - idx:                  moves horizontal to the column this thread is responsible for
      As[threadIdx.y][threadIdx.x] = A[idy*ds + (i * block_size) + threadIdx.x];
      Bs[threadIdx.y][threadIdx.x] = B[((i * block_size) + threadIdx.y)*ds + idx];

      // Synchronise, i.e. wait for the 32x32=1024 separate threads to have loaded
      // their elements into the shared memory so that the tiles are ready to use.
      __syncthreads();

      // Perform the dot product of row (from A) and column (from B) FOR THIS TILE,
      // keeping track of the running sum.
      for (int k = 0; k < block_size; k++)
      	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column

      // Synchronise, i.e. wait for the threads to complete their calculations, so that
      // the shared memory is not overwritten too early
      __syncthreads();

    }

    // Write to global memory
    C[idy*ds+idx] = temp;
  }
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;


  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  // start timing
  t0 = clock();

  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  for (int i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;}

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != A_val*B_val*DSIZE) {printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE); return -1;}
  printf("Success!\n"); 
  return 0;
}
  
