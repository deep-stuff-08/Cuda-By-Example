#include"common/book.h"

#define N (1024 * 1024)
#define DATA_SIZE (N * 20)

__global__ void kernel(int *a, int *b, int *c) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N) {
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}

int main(void) {
	cudaEvent_t start, stop;
	float elapsedTime;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	cudaStream_t stream0, stream1;
	HANDLE_ERROR(cudaStreamCreate(&stream0));
	HANDLE_ERROR(cudaStreamCreate(&stream1));

	int *host_a, *host_b, *host_c;
	int *dev_a0, *dev_b0, *dev_c0;
	int *dev_a1, *dev_b1, *dev_c1;

	HANDLE_ERROR(cudaMalloc((void**) &dev_a0, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_b0, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_c0, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_a1, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_b1, sizeof(int) * N));
	HANDLE_ERROR(cudaMalloc((void**) &dev_c1, sizeof(int) * N));

	HANDLE_ERROR(cudaHostAlloc((void**) &host_a, sizeof(int) * DATA_SIZE, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**) &host_b, sizeof(int) * DATA_SIZE, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**) &host_c, sizeof(int) * DATA_SIZE, cudaHostAllocDefault));

	for(int i = 0; i < DATA_SIZE; i++) {
		host_a[i] = rand();
		host_b[i] = rand();
	}

	HANDLE_ERROR(cudaEventRecord(start, 0));
	for(int i = 0; i < DATA_SIZE; i += N * 2) {
		HANDLE_ERROR(cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

		HANDLE_ERROR(cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

		kernel<<<N/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
		kernel<<<N/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);

		HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));	
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i  + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	HANDLE_ERROR(cudaStreamSynchronize(stream0));
	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time Taken: %3.1f\n", elapsedTime);

	HANDLE_ERROR(cudaFreeHost(host_a));
	HANDLE_ERROR(cudaFreeHost(host_b));
	HANDLE_ERROR(cudaFreeHost(host_c));

	HANDLE_ERROR(cudaFree(dev_a0));
	HANDLE_ERROR(cudaFree(dev_b0));
	HANDLE_ERROR(cudaFree(dev_c0));
	HANDLE_ERROR(cudaFree(dev_a1));
	HANDLE_ERROR(cudaFree(dev_b1));
	HANDLE_ERROR(cudaFree(dev_c1));

	HANDLE_ERROR(cudaStreamDestroy(stream0));
	HANDLE_ERROR(cudaStreamDestroy(stream1));
	return 0;
}