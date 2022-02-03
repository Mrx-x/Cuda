#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>

__global__ void add(float* a, float* b, float* c) {
		c[threadIdx.x + blockDim.x * blockIdx.x] = a[threadIdx.x + blockDim.x * blockIdx.x] + b[threadIdx.x + blockDim.x * blockIdx.x];
}

int main(void) {
	float* a, * b, * c; 
	float* dev_a, * dev_b, * dev_c; 
	int N = 1e6;

	cudaMalloc((void**)&dev_a, N * sizeof(float));
	cudaMalloc((void**)&dev_c, N * sizeof(float));
	cudaMalloc((void**)&dev_b, N * sizeof(float));

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
		auto start = std::chrono::steady_clock::now(); 
		add << < dim3(N / t), dim3(t) >> > (dev_a, dev_b, dev_c);
		cudaDeviceSynchronize();
		auto stop = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "time for ["<< t << "] = " << duration.count() << "(micro seconds)" << std::endl;	
		
	}
	cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	free(a); 
	free(b); 
	free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
