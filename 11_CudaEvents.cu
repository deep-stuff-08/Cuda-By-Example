#include<iostream>
#include"common/book.h"

#define N 65534

__global__ void addVec(int *a, int *b, int *c) {
	int tid = blockIdx.x;
	if(tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

int main(void) {
	cudaEvent_t start, stop;
	
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	HANDLE_ERROR(cudaMalloc((void**) &dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**) &dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**) &dev_c, N * sizeof(int)));

	for(int i = 1; i <= N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice));

	addVec<<<N, 1>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost));
	
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	
	for(int i = 0; i < N; i++) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	printf("time required to add two %d-sized vectors: %f\n", N, elapsedTime);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}