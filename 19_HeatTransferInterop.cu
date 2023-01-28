#include"common/book.h"
#include"common/gpu_anim.h"

#define DIM 1024
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED 0.25f

texture<float, 2> texConst;
texture<float, 2> texOut;
texture<float, 2> texIn;

__global__ void blend_kernel(float *dst, bool dstOut) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float t, l, c, r, b;
	if(dstOut) {
		t = tex2D(texIn, x, y+1);
		l = tex2D(texIn, x-1, y);
		c = tex2D(texIn, x, y);
		r = tex2D(texIn, x+1, y);
		b = tex2D(texIn, x, y-1);
	} else {
		t = tex2D(texOut, x, y+1);
		l = tex2D(texOut, x-1, y);
		c = tex2D(texOut, x, y);
		r = tex2D(texOut, x+1, y);
		b = tex2D(texOut, x, y-1);	
	}
	dst[offset] = c + SPEED * (l + t + r + b - c * 4);
}


__global__ void copy_const_kernel(float *iptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex2D(texConst, x, y);
	if(c != 0) {
		iptr[offset] = c;
	}
}

struct DataBlock {
	float *dev_in;
	float *dev_out;
	float *dev_const;

	cudaEvent_t start, stop;

	float totalTimes;
	float frames;
};

void anim_gpu(uchar4 *ptr, DataBlock* d, int ticks) {
	HANDLE_ERROR(cudaEventRecord(d->start, 0));
	dim3 blocks(DIM/16, DIM/16);
	dim3 threads(16, 16);

	volatile bool dstOut = true;
	for(int i = 0; i < 90; i++) {
		float *in, *out;
		if(dstOut) {
			in = d->dev_in;
			out = d->dev_out;
		} else {
			in = d->dev_out;
			out = d->dev_in;
		}
		copy_const_kernel<<<blocks, threads>>>(in);
		blend_kernel<<<blocks, threads>>>(out, dstOut);
		dstOut = !dstOut;
	}
	float_to_color<<<blocks, threads>>>(ptr, d->dev_in);
	HANDLE_ERROR(cudaEventRecord(d->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(d->stop));
	float time;
	HANDLE_ERROR(cudaEventElapsedTime(&time, d->start, d->stop));
	d->totalTimes += time;
	d->frames++;

	printf("Average Time per Frame %3.1f ms\n", d->totalTimes / d->frames);
}

void anim_exit(DataBlock *d) {
	HANDLE_ERROR(cudaUnbindTexture(texIn));
	HANDLE_ERROR(cudaUnbindTexture(texOut));
	HANDLE_ERROR(cudaUnbindTexture(texConst));
	HANDLE_ERROR(cudaFree(d->dev_in));
	HANDLE_ERROR(cudaFree(d->dev_out));
	HANDLE_ERROR(cudaFree(d->dev_const));
	HANDLE_ERROR(cudaEventDestroy(d->start));
	HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void) {
	DataBlock data;
	GPUAnimBitmap bitmap(DIM, DIM, &data);

	data.totalTimes = 0;
	data.frames = 0;

	HANDLE_ERROR(cudaEventCreate(&data.start));
	HANDLE_ERROR(cudaEventCreate(&data.stop));

	int img_size = bitmap.image_size();

	HANDLE_ERROR(cudaMalloc((void**) &data.dev_in, img_size));
	HANDLE_ERROR(cudaMalloc((void**) &data.dev_out, img_size));
	HANDLE_ERROR(cudaMalloc((void**) &data.dev_const, img_size));

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	HANDLE_ERROR(cudaBindTexture2D(NULL, texIn, data.dev_in, desc, DIM, DIM, sizeof(float) * DIM));
	HANDLE_ERROR(cudaBindTexture2D(NULL, texOut, data.dev_out, desc, DIM, DIM, sizeof(float) * DIM));
	HANDLE_ERROR(cudaBindTexture2D(NULL, texConst, data.dev_const, desc, DIM, DIM, sizeof(float) * DIM));

	float *temp = (float*)malloc(img_size);
	for(int i = 0; i < DIM * DIM; i++) {
		temp[i] = 0;
		int x = i % DIM;
		int y = i / DIM;
		if((x > 300) && (x < 600) && (y > 310) && (y < 601)) {
			temp[i] = MAX_TEMP;
		}
	}

	temp[DIM*100+100] = (MAX_TEMP + MIN_TEMP) / 2;
	temp[DIM*700+100] = MIN_TEMP;
	temp[DIM*300+300] = MIN_TEMP;
	temp[DIM*200+700] = MIN_TEMP;

	for(int y = 800; y < 900; y++) {
		for(int x = 400; x < 500; x++) {
			temp[x+y*DIM] = MIN_TEMP;
		}
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_const, temp, img_size, cudaMemcpyHostToDevice));

	for(int y = 800; y < DIM; y++) {
		for(int x = 0; x < 200; x++) {
			temp[x+y*DIM] = MAX_TEMP;
		}
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_in, temp, img_size, cudaMemcpyHostToDevice));
	free(temp);
	bitmap.anim_and_exit((void (*) (uchar4*, void*, int))anim_gpu, (void (*)(void*))anim_exit);
}