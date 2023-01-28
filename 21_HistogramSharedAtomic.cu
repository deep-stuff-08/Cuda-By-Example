#include"common/book.h"

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char* buffer, long size, unsigned int* histo) {
	__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while(i < size) {
		atomicAdd(&temp[buffer[i]], 1);
		i += stride;
	}

	__syncthreads();
	atomicAdd(&histo[buffer[threadIdx.x]], temp[threadIdx.x]);
}

int main(void) {
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE);
	unsigned int histo[256];
	memset(histo, 0, 256);

	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	unsigned char* dev_buffer;
	unsigned int* dev_histo;
	HANDLE_ERROR(cudaMalloc((void**) &dev_buffer, SIZE));
	HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMalloc((void**) &dev_histo, sizeof(int) * 256));
	HANDLE_ERROR(cudaMemset(dev_histo, 0, sizeof(int) * 256));

	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	int blocks = prop.multiProcessorCount * 2;
	histo_kernel<<<blocks, 256>>>(dev_buffer, SIZE, dev_histo);

	HANDLE_ERROR(cudaMemcpy(histo, dev_histo, sizeof(int) * 256, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float time;
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	printf("Time required = %3.1f ms\n", time);

	long histoCount = 0;
	for(int i = 0; i < 256; i++) {
		histoCount += histo[i];
	}
	if(histoCount != SIZE) {
		printf("Histogram Failed\n");
		for(int i = 0; i < SIZE; i++) {
			histo[buffer[i]]--;
		}
		for(int i = 0; i < 256; i++) {
			if(histo[i] != 0) {
				printf("Failure At %d\n", i+1);
			}
		}
	}
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	HANDLE_ERROR(cudaFree(dev_buffer));
	HANDLE_ERROR(cudaFree(dev_histo));
	free(buffer);
}