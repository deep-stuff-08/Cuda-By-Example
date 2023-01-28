#include<iostream>
#include<stdlib.h>
#include"common/book.h"

#define N 33553920

__global__ void vecAdd(int *a, int *b, int *c) {
	long tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < N) {
		c[tid] = a[tid] + b[tid];
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
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

	vecAdd<<<(N + 511) / 512, 512>>>(dev_a, dev_b, dev_c);
	
	HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));
	for(int i = 0; i < N; i+=10) {
		printf("No: %ld -> %d + %d = %d\n", i, a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}