#include"common/book.h"
#include"common/cpu_bitmap.h"

#define DIM 1024
#define PI 3.1415926535897932

__global__ void kernel(unsigned char* bitmap) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	__shared__ float shared[16][16];
	const float period = 128.0f;

	shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI/ period) + 1.0f) * (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

	bitmap[offset * 4 + 0] = 0;
	bitmap[offset * 4 + 1] = shared[15 - threadIdx.x][ 15 -threadIdx.y];
	bitmap[offset * 4 + 2] = 0;
	bitmap[offset * 4 + 3] = 255;
}

__global__ void kernelSync(unsigned char* bitmap) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	__shared__ float shared[16][16];
	const float period = 128.0f;

	shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI/ period) + 1.0f) * (sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

	__syncthreads();

	bitmap[offset * 4 + 0] = 0;
	bitmap[offset * 4 + 1] = shared[15 - threadIdx.x][ 15 -threadIdx.y];
	bitmap[offset * 4 + 2] = 0;
	bitmap[offset * 4 + 3] = 255;
}

int main(int argv, char** argc) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**) &dev_bitmap, bitmap.image_size()));

	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16, 16);

	printf("Use Thread Syncing?(y/n):");
	char c = getchar();

	if(c == 'y') {
		kernelSync<<<grids, threads>>>(dev_bitmap);
	} else if(c == 'n') {
		kernel<<<grids, threads>>>(dev_bitmap);
	} else {
		printf("Defaulting to 'n'\n");
		kernel<<<grids, threads>>>(dev_bitmap);
	}
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
}