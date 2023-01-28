#include<iostream>
#include<stdlib.h>
#include<unistd.h>
#include"common/book.h"


#define N 300000000

__global__ void reallyLongVecAdd(int *a, int *b, int *c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	while(tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

int main(void) {
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;

	a = (int*)malloc(sizeof(int) * N);
	b = (int*)malloc(sizeof(int) * N);
	c = (int*)malloc(sizeof(int) * N);

	HANDLE_ERROR(cudaMalloc((void**) &dev_a, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_b, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_c, sizeof(int) * N));

	for(int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = N - i;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

	reallyLongVecAdd<<<128, 128>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

	for(int i = 0; i < N; i += 100) {
		printf("No. %d -> %d + %d = %d\n", i, a[i], b[i], c[i]);
	}
	sleep(100000000);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}