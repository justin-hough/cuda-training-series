/*
The tutorial refers to what I call convolutions (in a CV context) as stencil operations.
This file includes a 1D stencil (1D-conv) kernel that utilises shared memory in the threads
within a block.
*/

#include <stdio.h>
#include <algorithm>

using namespace std;

#define N 4096 // The size of the input array
#define RADIUS 3 // The 1D stencil's width is 2R+1 i.e. 7
#define BLOCK_SIZE 16 // Number of threads per block

__global__ void stencil_1d(int *in, int *out) {
    __shared__ int temp[BLOCK_SIZE + 2*RADIUS]; // Shared memory for this block's section of the input (padded by the stencil radius)
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; // Global index, unique within the entire grid
    int lindex = threadIdx.x + RADIUS; // Local index, unique within the block

    // Read input elements into shared memory
    temp[lindex] = in[gindex]; // Store the main element in shared memor
    if (threadIdx.x < RADIUS) { // For the thread's within the radius, also store:
      temp[lindex - RADIUS] = in[gindex - RADIUS]; // the elements to the left of the main data
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE]; // the elements to the right of the main data
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
      result += temp[lindex + offset];

    // Store the result
    out[gindex] = result;
}

void fill_ints(int *x, int n) {
  fill_n(x, n, 1);
}

int main(void) {
  int *in, *out; // host copies of a, b, c
  int *d_in, *d_out; // device copies of a, b, c

  // Alloc space for host copies and setup values
  // NOTE: These arrays INCLUDE the padding
  int size = (N + 2*RADIUS) * sizeof(int);
  in = (int *)malloc(size); fill_ints(in, N + 2*RADIUS);
  out = (int *)malloc(size); fill_ints(out, N + 2*RADIUS);

  // Alloc space for device copies
  cudaMalloc((void **)&d_in, size); 
  cudaMalloc((void **)&d_out, size);

  // Copy to device
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

  // Launch stencil_1d() kernel on GPU
  stencil_1d<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);

  // Copy result back to host
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  // Error Checking
  for (int i = 0; i < N + 2*RADIUS; i++) {
    if (i<RADIUS || i>=N+RADIUS){
      if (out[i] != 1)
    	printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
    } else {
      if (out[i] != 1 + 2*RADIUS)
    	printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2*RADIUS);
    }
  }

  // Cleanup
  free(in); free(out);
  cudaFree(d_in); cudaFree(d_out);
  printf("Success!\n");
  return 0;
}
