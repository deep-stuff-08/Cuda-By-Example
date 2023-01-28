#include"common/book.h"
#include"common/cpu_bitmap.h"
#include<time.h>

#define rnd(x) (x * rand() / RAND_MAX)
#define SPHERES 20
#define INF 2e10f
#define DIM 1024

typedef struct Sphere {
	float x, y, z;
	float r, g, b;
	float radius;

	__device__ float hit(float ox, float oy, float *n) {
		float dx = ox - x;
		float dy = oy - y;

		if(dx*dx + dy*dy < radius*radius) {
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
} Sphere;

__constant__ Sphere dev_s[SPHERES];

__global__ void kernel(unsigned char* ptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = (x - DIM/2);
	float oy = (y - DIM/2);

	float r = 0.0f, g = 0.0f, b = 0.0f;
	float maxz = -INF;

	for(int i = 0; i < SPHERES; i++) {
		float n;
		float t = dev_s[i].hit(ox, oy, &n);
		if(t > maxz) {
			r = dev_s[i].r * n;
			g = dev_s[i].g * n;
			b = dev_s[i].b * n;
			maxz = t;
		}
	}

	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;
}

int main(void) {
	srand(time(0));
	cudaEvent_t start, stop;
	
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;
	HANDLE_ERROR(cudaMalloc((void**) &dev_bitmap, bitmap.image_size()));
	
	Sphere* temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	
	for(int i = 0; i < SPHERES; i++) {
		temp_s[i].x = rnd(1000.0f) - 500.0f;
		temp_s[i].y = rnd(1000.0f) - 500.0f;
		temp_s[i].z = rnd(1000.0f) - 500.0f;

		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);

		temp_s[i].radius = rnd(100.0f) + 20.0f;
	}

	HANDLE_ERROR(cudaMemcpyToSymbol(dev_s, temp_s, sizeof(Sphere) * SPHERES));
	free(temp_s);

	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16, 16);
	kernel<<<grids, threads>>>(dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	printf("time required for ray-tracing with constant memory: %f\n", elapsedTime);
	
	bitmap.display_and_exit();

	cudaFree(dev_bitmap);
}