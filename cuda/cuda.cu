#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <string.h>

// CUDA Runtime
#include <cuda_runtime.h>

//#define TEST
#ifdef TEST

#include "../assets/test_image.h" // 2 048 pixels
#define NBR_OF_ELEMENTS 2048	  // width * height in 64x32 test image.

#else

#define NBR_OF_ELEMENTS 2073600 // width * height in FullHD. Change for bigger images!
#include "../assets/image.h"	// 2 073 600 pixels

#endif

#define HISTOGRAM_SIZE (255 * 3 + 1) // max color value

void load_image(unsigned int *image_data)
{
	char pixel[3];
	char *data = (char *)header_data;

	for (unsigned long i = 0L; i < NBR_OF_ELEMENTS; ++i)
	{
		HEADER_PIXEL(data, pixel);
		image_data[i] = (uint8_t)pixel[0] + (uint8_t)pixel[1] + (uint8_t)pixel[2];
	}
}

void print_image(unsigned int *image_data)
{
	printf("Image printing:\n");

	for (unsigned long i = 0L; i < NBR_OF_ELEMENTS; ++i)
	{
		printf("%u, ", image_data[i]);
	}

	printf("\n");
}

void print_histogram(unsigned long *histogram)
{
	if (!histogram)
	{
		printf("Histogram has no values!");
		return;
	}

	printf("Histogram values:\n");

	for (unsigned int i = 0u; i < HISTOGRAM_SIZE; ++i)
	{
		if (0L != histogram[i])
		{
			printf("v: %u, a: %lu\n", i, histogram[i]);
		}
	}

	printf("\n");
}

void clear_histogram(unsigned long *histogram)
{
	if (!histogram)
	{
		printf("Histogram has no values!");
		return;
	}
	for (unsigned long i = 0L; i < HISTOGRAM_SIZE; ++i)
	{
		histogram[i] = 0L;
	}
}

int are_histograms_equal(unsigned long *array, unsigned long *array2)
{
	for (unsigned int i = 0u; i < HISTOGRAM_SIZE; ++i)
	{
		if (array[i] != array2[i])
		{
			return -1;
		}
	}
	return 0;
}

__global__ void calculations(unsigned int *image_data, unsigned int *histogram, unsigned long *i)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int N = blockDim.x * gridDim.x;
	int pixels_per_thread = NBR_OF_ELEMENTS / (N - 1);
	int last_thread_pixels = NBR_OF_ELEMENTS % (N - 1);

	// Calculate
	if (index < N - 1)
	{
		for (int j = 0; j < pixels_per_thread; ++j)
		{
			atomicAdd(&(histogram[image_data[index * pixels_per_thread + j]]), 1);
		}
	}
	else
	{
		for (int j = 0; j < last_thread_pixels; ++j)
		{
			atomicAdd(&(histogram[image_data[index * pixels_per_thread + j]]), 1);
		}
	}

	*i = N;
}

int main(int argc, char **argv)
{
	int nBlk = 512;
	int nThx = 512;
	int N = nBlk * nThx;

	size_t size_image = NBR_OF_ELEMENTS * sizeof(unsigned int);
	size_t size_histogram = HISTOGRAM_SIZE * sizeof(unsigned long);

	int pixels_per_thread = NBR_OF_ELEMENTS / (N - 1);
	while (pixels_per_thread < 8)
	{
		nBlk >>= 1;
		nThx >>= 1;
		N = nBlk * nThx;
		pixels_per_thread = NBR_OF_ELEMENTS / (N - 1);
	}
	int last_thread_pixels = NBR_OF_ELEMENTS - pixels_per_thread * (N - 1);
	printf("There will be nBlk: %d, nThx: %d, which gives ", nBlk, nThx);
	printf("%d pixels per thread and %d pixels as a reminder for the last one.\n",
		   pixels_per_thread, last_thread_pixels);

	unsigned long *histogram, *d_histogram, *histogram_result;
	unsigned int *image_data, *d_image_data;
	unsigned long *x = 0UL, *d_x;

	// Alloc space for device copies od input and output data
	printf("Allocation of space for device copies\n");
	cudaMalloc((void **)&d_image_data, size_image);
	cudaMalloc((void **)&d_histogram, size_histogram);
	cudaMalloc((void **)&d_x, sizeof(x));

	// Alloc space for host copies of input, output data and setup input values
	printf("Allocation of space for host copies\n");
	image_data = (unsigned int *)malloc(size_image);
	load_image(image_data);
	histogram = (unsigned long *)malloc(size_histogram);
	clear_histogram(histogram);
	histogram_result = (unsigned long *)malloc(size_histogram);
	clear_histogram(histogram_result);

	x = (unsigned long *)malloc(sizeof(x));
	*x = 0UL;
	// Copy data to device
	printf("Copy data to device\n");
	cudaMemcpy(d_image_data, image_data, size_image, cudaMemcpyHostToDevice);
	cudaMemcpy(d_histogram, histogram, size_histogram, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, sizeof(x), cudaMemcpyHostToDevice);

	// Lunch calculations() kernel on GPU with N threads
	printf("Calculations\n");
	calculations<<<nBlk, nThx>>>(d_image_data, (unsigned int *)d_histogram, d_x);

	// Copy result back to host
	printf("Copy result to host\n");
	cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);
	cudaMemcpy(x, d_x, sizeof(x), cudaMemcpyDeviceToHost);
	printf("x: %lu\n", *x);

	// Checkup
	printf("Checkup\n");
	int i;
	for (i = 0; i < N; ++i)
	{
		if (i < N - 1)
		{
			for (int j = 0; j < pixels_per_thread; ++j)
			{
				++histogram_result[image_data[i * pixels_per_thread + j]]; // thread idx * pixels per thread + pixel idx per thread
			}
		}
		else
		{
			for (int j = 0; j < last_thread_pixels; ++j)
			{
				++histogram_result[image_data[i * pixels_per_thread + j]]; // thread idx * pixels per thread + pixel idx per thread
			}
		}
	}

	// Checking correctness
	printf("Checking correctness\n");
	if (are_histograms_equal(histogram, histogram_result) != 0)
	{
		printf("Histograms are not equal!\n");
		print_histogram(histogram);
		//print_histogram(histogram_result);
	}
	else
	{
		printf("Success!\n");
		//print_histogram(histogram);
		//print_histogram(histogram_result);
	}

	// Cleanup
	printf("Cleanup\n");
	free(histogram);
	free(image_data);
	free(histogram_result);
	free(x);
	cudaFree(d_histogram);
	cudaFree(d_image_data);
	cudaFree(d_x);

	//system("pause");
	return 0;
}
