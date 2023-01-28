#include"common/book.h"
#include"common/gpu_anim.h"

#define DIM 1024

void __global__ kernel(uchar4 *ptr, int ticks) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x - DIM/2;
	float fy = y - DIM/2;
	float d = sqrtf(fx * fx + fy * fy);

	unsigned char blue = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));    
    ptr[offset].x = 0;
	ptr[offset].y = 0;
	ptr[offset].z = blue;
	ptr[offset].w = 255;
}

void generateFrame(uchar4 *ptr, void*, int ticks) {
	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16, 16);
	kernel<<<grids, threads>>>(ptr, ticks);
}

int main(void) {
	GPUAnimBitmap bitmap(DIM, DIM, NULL);
	bitmap.anim_and_exit((void (*)(uchar4*, void*, int))generateFrame, NULL);
}
