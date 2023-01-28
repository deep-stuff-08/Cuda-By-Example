#include"common/book.h"

#include"common/book.h"

#define iMin(a, b) (a < b)?a:b

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = iMin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

struct DataStruct {
	int devId;
	int size;
	int offset;
	float *a;
	float *b;
	float retValue;
};

__global__ void dot(int size, float *a, float *b, float *c) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float temp = 0;
	while(tid < size) {
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

void *routine(void *pvoiddata) {
	DataStruct *data = (DataStruct*)pvoiddata;
	if(data->devId != 0) {
		HANDLE_ERROR(cudaSetDevice(data->devId));
		HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
	}
	
	int size = data->size;
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;

	a = data->a;
	b = data->b;
	partial_c = (float*)malloc(sizeof(float) * blocksPerGrid);

	HANDLE_ERROR(cudaHostGetDevicePointer(&dev_a, a, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer(&dev_b, b, 0));
	HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, sizeof(float) * blocksPerGrid));

	dev_a += data->offset;
	dev_b += data->offset;

	dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

	HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost));

	c = 0;
	for(int i = 0; i < blocksPerGrid; i++) {
		c += partial_c[i];
	}

	data->retValue = c;

	cudaFree(dev_partial_c);

	free(partial_c);

	return 0;
}

int main(void) {
	float *a, *b;
	
	int devCount;
	HANDLE_ERROR(cudaGetDeviceCount(&devCount));
	if(devCount < 2) {
		printf("Not enough CUDA capable devices found.\n");
		return 0;
	}

	HANDLE_ERROR(cudaSetDevice(0));
	HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
	HANDLE_ERROR(cudaHostAlloc((void**) &a, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**) &b, N * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocPortable | cudaHostAllocMapped));

	for(int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i * 2;
	}

	DataStruct data[2];
	data[0].devId = 0;
	data[0].size = N / 2;
	data[0].offset = 0;
	data[0].a = a;
	data[0].b = b;

	data[1].devId = 1;
	data[1].size = N / 2;
	data[1].offset = N / 2;
	data[1].a = a + N/2;
	data[1].b = b + N/2;

	CUTThread thread = start_thread(routine, data);
	routine(data+1);
	end_thread(thread);

	HANDLE_ERROR(cudaFreeHost(a));
	HANDLE_ERROR(cudaFreeHost(b));

	printf("Value calculated: %f\n", data[0].retValue + data[1].retValue);
	return 0;
}