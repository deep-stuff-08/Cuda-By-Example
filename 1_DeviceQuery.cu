#include<iostream>
#include"common/book.h"

int main(void) {
	cudaDeviceProp prop;
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for(int i = 0; i < count; i++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf("-------------------Information For Device %d--------------------\n", i);
		printf("**********************General Information**********************\n");
		printf("Graphic Processor Name:\t\t%s\n", prop.name);
		printf("Compute Capability:\t\t%d.%d\n", prop.major, prop.minor);
		printf("Clock Rate:\t\t\t%d MHz\n", prop.clockRate/1000);
		printf("Device Copy Overlap:\t\t%s\n", (prop.deviceOverlap)?"Enabled":"Disabled");
		printf("Kernel Execution Timeout:\t%s\n", (prop.kernelExecTimeoutEnabled)?"Enabled":"Disabled");
		printf("Graphic Card Type:\t\t%s\n\n", (prop.integrated)?"Integrated":"Discrete");
		printf("***********************Memory Information**********************\n");
		printf("Total Global Memory:\t\t%.2lf GB\n", (double)prop.totalGlobalMem/1024/1024/1024);
		printf("Total Constant Memory:\t\t%.0lf KB\n", (double)prop.totalConstMem/1024);
		printf("Max Memory Pitch:\t\t%.0lf MB\n", (double)prop.memPitch/1024/1024);
		printf("Texture Alignment:\t\t%ld\n\n", prop.textureAlignment);
		printf("******************Multi Processor Information******************\n");
		printf("Multi Processor Count:\t\t%d\n", prop.multiProcessorCount);
		printf("Shared Memory Per MP:\t\t%.0lf KB\n", (double)prop.sharedMemPerBlock/1024);
		printf("Registers Per MP:\t\t%d\n", prop.regsPerBlock);
		printf("Threads in Warp:\t\t%d\n", prop.warpSize);
		printf("Max Threads Per Block:\t\t%d\n", prop.maxThreadsPerBlock);
		printf("Max Thread Dimensions:\t\t(%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max Grid Dimensions:\t\t(%d, %d, %d)\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}
}