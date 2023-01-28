#include"common/book.h"
#include"common/cpu_anim.h"

#define DIM 1024
#define PI 3.1415926535897932
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25001f

struct DataBlock {
	unsigned char* op_bitmap;
	float* dev_InSrc;
	float* dev_OutSrc;
	float* dev_ConstSrc;
	CPUAnimBitmap *bitmap;
	cudaEvent_t start, stop;
	float totalTimes;
	float frames;
};

texture<float, 2> texIn;
texture<float, 2> texOut;
texture<float, 2> texConst;

__global__ void copy_const_kernel(float* iPtr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex2D(texConst, x, y);
	if(c != 0) {
		iPtr[offset] = c;
	}
}

__global__ void blend_kernel(float *dst, bool dstOut) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float t, l, c, r, b;
	if(dstOut) {
		t = tex2D(texIn, x, y-1);
		l = tex2D(texIn, x-1, y);
		c = tex2D(texIn, x, y);
		r = tex2D(texIn, x+1, y);
		b = tex2D(texIn, x, y+1);
	} else {
		t = tex2D(texOut, x, y-1);
		l = tex2D(texOut, x-1, y);
		c = tex2D(texOut, x, y);
		r = tex2D(texOut, x+1, y);
		b = tex2D(texOut, x, y+1);
	}
	dst[offset] = c + SPEED * (t + b + l + r - c * 4);
}

void anim_gpu(DataBlock *d, int ticks) {
	HANDLE_ERROR(cudaEventRecord(d->start, 0));
	dim3 blocks(DIM/16, DIM/16);
	dim3 threads(16, 16);
	CPUAnimBitmap *bitmap = d->bitmap;

	volatile bool dstOut = true;
	for(int i = 0; i < 90; i++) {
		float* in,* out;
		if(dstOut) {
			in = d->dev_InSrc;
			out = d->dev_OutSrc;
		} else {
			in = d->dev_OutSrc;
			out = d->dev_InSrc;
		}
		copy_const_kernel<<<blocks, threads>>>(in);
		blend_kernel<<<blocks, threads>>>(out, dstOut);
		dstOut = ! dstOut;
	}
	float_to_color<<<blocks, threads>>>(d->op_bitmap, d->dev_InSrc);
	HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->op_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaEventRecord(d->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(d->stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));
	d->totalTimes += elapsedTime;
	++d->frames;
	printf("Average Time per Frame: %3.1f ms\n", d->totalTimes / d->frames);
}

void anim_exit(DataBlock *d) {
	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConst);

	cudaFree(d->dev_InSrc);
	cudaFree(d->dev_OutSrc);
	cudaFree(d->dev_ConstSrc);

	HANDLE_ERROR(cudaEventDestroy(d->start));
	HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void) {
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;
	data.totalTimes = 0;
	data.frames = 0;
	HANDLE_ERROR(cudaEventCreate(&data.start));
	HANDLE_ERROR(cudaEventCreate(&data.stop));

	HANDLE_ERROR(cudaMalloc((void**) &data.op_bitmap, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**) &data.dev_InSrc, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**) &data.dev_OutSrc, bitmap.image_size()));
	HANDLE_ERROR(cudaMalloc((void**) &data.dev_ConstSrc, bitmap.image_size()));

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTexture2D(NULL, texConst, data.dev_ConstSrc, desc, DIM, DIM, sizeof(float) * DIM));
	HANDLE_ERROR(cudaBindTexture2D(NULL, texIn, data.dev_InSrc, desc, DIM, DIM, sizeof(float) * DIM));
	HANDLE_ERROR(cudaBindTexture2D(NULL, texOut, data.dev_OutSrc, desc, DIM, DIM, sizeof(float) * DIM));

	float *temp = (float*)malloc(bitmap.image_size());
	for(int i = 0; i < DIM * DIM; i++) {
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if((x > 300) && (x < 600) && (y > 310) && (y < 601)) {
			temp[i] = MAX_TEMP;
		}
	}
	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;
	for(int y = 800; y < 900; y++) {
		for(int x = 400; x < 500; x++) {
			temp[x + y * DIM] = MIN_TEMP;
		}
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_ConstSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice));
	free(temp);

	bitmap.anim_and_exit((void (*)(void*, int))anim_gpu, (void (*)(void*))anim_exit);
}