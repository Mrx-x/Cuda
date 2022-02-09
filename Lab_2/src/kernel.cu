#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

#define CUDA_CHECK_RETURN(value){\
	cudaError_t _m_cudaStat = value;\
	if(_m_cudaStat != cudaSuccess){\
		fprintf(stderr, "ERROR %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}}

__global__ void add(float* a, float* b, float* c) {
	c[threadIdx.x + blockDim.x * blockIdx.x] = a[threadIdx.x + blockDim.x * blockIdx.x] + b[threadIdx.x + blockDim.x * blockIdx.x];
}

int main(void) {
	float* a, * b, * c;
	float* dev_a, * dev_b, * dev_c;
	int N = 1e6;
	float elapsedTime;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int countDevice;
	cudaGetDeviceCount(&countDevice);
	if (countDevice == 0) {
		fprintf(stderr, "[ERROR] - There is no device.\n");
	}
	else printf("Count device == [%d]\n", countDevice);

	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);	

	printf("Device name: %s.\n", deviceProp.name);
	printf("Total Global Memory = %lu bytes.\n", deviceProp.totalGlobalMem);
	printf("Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
	printf("Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
	printf("Total number of registers available per block: %d \n", deviceProp.regsPerBlock);
	printf("Warp size: %d\n", deviceProp.warpSize);
	printf("Max grid size: %lu.\n", deviceProp.maxGridSize);
	printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);

	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_a, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_c, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_b, N * sizeof(float)));

	a = (float*)calloc(N, sizeof(float));
	b = (float*)calloc(N, sizeof(float));
	c = (float*)calloc(N, sizeof(float));

	for (int i = 0; i < N; ++i)
	{
		a[i] = (float)rand() / (float)RAND_MAX;
		b[i] = (float)rand() / (float)RAND_MAX;
	}

	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

	for (int t = 2; t <= 1024; t *= 2)
	{
		cudaEventRecord(start, 0);
		add << < dim3(N / t), dim3(t) >> > (dev_a, dev_b, dev_c);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);

		fprintf(stderr, "gTest took %g per milliseconds\n", elapsedTime);
	}
	CUDA_CHECK_RETURN(cudaGetLastError());
	CUDA_CHECK_RETURN(cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost));


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(a);
	free(b);
	free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}