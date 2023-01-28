#include<iostream>
#include"common/book.h"

#define N 512

__global__ void vecAdd(int *a, int *b, int *c) {
	int tid = threadIdx.x;
	if(tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	HANDLE_ERROR(cudaMalloc((void**) &dev_a, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_b, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_c, sizeof(int) * N));

	for(int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

	vecAdd<<<1, N>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));

	for(int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_c));
}
