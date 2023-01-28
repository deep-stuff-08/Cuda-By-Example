#include"common/book.h"

float check_pageable_memory_time(int size, int times, bool up) {
	cudaEvent_t start, stop;
	float time;
	char *a, *dev_a;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	a = (char*)malloc(size);
	HANDLE_NULL(a);
	HANDLE_ERROR(cudaMalloc((void**) &dev_a, size));

	HANDLE_ERROR(cudaEventRecord(start, 0));
	for(int i = 0; i < times; i++) {
		if(up) {
			HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
		} else {
			HANDLE_ERROR(cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost));
		}
	}
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	
	free(a);
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	return time;
}

float check_pinned_memory_time(int size, int times, bool up) {
	cudaEvent_t start, stop;
	float time;
	char *a, *dev_a;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaHostAlloc((void**) &a, size, cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**) &dev_a, size));

	HANDLE_ERROR(cudaEventRecord(start, 0));
	for(int i = 0; i < times; i++) {
		if(up) {
			HANDLE_ERROR(cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice));
		} else {
			HANDLE_ERROR(cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost));
		}
	}
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));
	
	HANDLE_ERROR(cudaFreeHost(a));
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	
	return time;
}

int main(int argc, char** argv) {
	int size = 1024;
	int time = 5;
	float time1, time2, time3, time4;
	printf("Memory Size: %d MB\n", size);
	printf("Epochs     : %d\n", time);
	printf("Running Host To Device For Pageable Memory...\n");
	time1 = check_pageable_memory_time(size * 1024 * 1024, time, true) / 1000;
	printf("Running Device To Host For Pageable Memory...\n");
	time2 = check_pageable_memory_time(size * 1024 * 1024, time, false) / 1000;
	printf("Running Host To Device For Pinned Memory...\n");
	time3 = check_pinned_memory_time(size * 1024 * 1024, time, true) / 1000;
	printf("Running Device To Host For Pinned Memory...\n");
	time4 = check_pinned_memory_time(size * 1024 * 1024, time, false) / 1000;

	size *= time;
	printf("\nChecking Pageable Memory Speeds:\n");
	printf("\tHost To Device\n");
	printf("\t\tTime Required: %3.5f sec\n", time1);
	printf("\t\tTransfer Rate: %3.1f MB/s\n", size / time1);
	printf("\tDevice To Host\n");
	printf("\t\tTime Required: %3.5f sec\n", time2);
	printf("\t\tTransfer Rate: %3.1f MB/s\n", size / time2);
	printf("\nChecking Pinned Memory Speeds:\n");
	printf("\tHost To Device\n");
	printf("\t\tTime Required: %3.5f sec\n", time3);
	printf("\t\tTransfer Rate: %3.1f MB/s\n", size / time3);
	printf("\tDevice To Host\n");
	printf("\t\tTime Required: %3.5f sec\n", time4);
	printf("\t\tTransfer Rate: %3.1f MB/s\n", size / time4);
}