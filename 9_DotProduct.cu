#include"common/book.h"

#define iMin(a, b) (a < b)?a:b

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = iMin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while(tid < N) {
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = temp;

	__syncthreads();

	int i = blockDim.x / 2;
	while(i != 0) {
		if(cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + 1];
		}
		__syncthreads();
		i /= 2;
	}

	if(cacheIndex == 0) {
		c[blockIdx.x] = cache[0];
	}
}

int main(void) {
	float *a, *b, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;
	float c;

	a = (float*)malloc(sizeof(float) * N);
	b = (float*)malloc(sizeof(float) * N);
	partial_c = (float*)malloc(sizeof(float) * blocksPerGrid);

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, sizeof(float) * N));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, sizeof(float) * N));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, sizeof(float) * blocksPerGrid));

	for(int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, sizeof(float) * N, cudaMemcpyHostToDevice));

	dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost));

	c = 0;
	for(int i = 0; i < blocksPerGrid; i++) {
		c += partial_c[i];
	}

	printf("Dot product = %g\n", c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	free(a);
	free(b);
	free(partial_c);
}