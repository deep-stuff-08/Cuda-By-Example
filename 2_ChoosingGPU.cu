#include<iostream>
#include"common/book.h"

int main(void) {
	cudaDeviceProp prop;
	int dev;
	HANDLE_ERROR(cudaGetDevice(&dev));
	printf("Current Device: %d\n", dev);
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 6;
	prop.minor = 5;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	printf("Closest Device: %d\n", dev);
	HANDLE_ERROR(cudaSetDevice(dev));
}
