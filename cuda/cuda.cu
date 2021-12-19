#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <string.h>


#define TEST
#ifdef TEST

#include "../assets/test_image.h" // 2 048 pixels
#define NBR_OF_ELEMENTS 2048	  // width * height in 64x32 test image.

#else

#define NBR_OF_ELEMENTS 2073600 // width * height in FullHD. Change for bigger images!
#include "../assets/image.h"	// 2 073 600 pixels

#endif

#define HISTOGRAM_SIZE 255 * 3 + 1 // max color value


void load_image(unsigned int * image_data)
{
	char pixel[3];
	char *data = (char *)header_data;

	for (unsigned long i = 0L; i < NBR_OF_ELEMENTS; ++i)
	{
		HEADER_PIXEL(data, pixel);
		image_data[i] = (uint8_t)pixel[0] + (uint8_t)pixel[1] + (uint8_t)pixel[2];
	}
}

void print_image(unsigned int * image_data)
{
	printf("Image printing:\n");

	for (unsigned long i = 0L; i < NBR_OF_ELEMENTS; ++i)
	{
		printf("%u, ", image_data[i]);
	}

	printf("\n");
}

void print_histogram(unsigned long * histogram)
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

void clear_histogram(unsigned long * histogram)
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

bool are_histograms_equal(unsigned long * array, unsigned long * array2)
{
	for (unsigned int i = 0u; i < HISTOGRAM_SIZE; ++i)
	{
		if(array[i] != array2[i])
		{
			return false;
		}
	}
	return true;
}

__global__ void calculations(unsigned int * image_data, unsigned long * histogram)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int j;
	int N = blockDim.x * gridDim.x;
	int pixels_per_thread = NBR_OF_ELEMENTS / (N - 1);
	int last_thread_pixels = NBR_OF_ELEMENTS - pixels_per_thread * (N - 1);

__shared__ unsigned int tmp_image_data[NBR_OF_ELEMENTS];
__shared__ unsigned long tmp_histogram[HISTOGRAM_SIZE];

	// Copy to tmp
	for(j = 0; j < pixels_per_thread * blockDim.x; ++j)
	{
		if(blockIdx.x * blockDim.x * pixels_per_thread + j < NBR_OF_ELEMENTS)
		{
			tmp_image_data[j] = image_data[blockIdx.x * blockDim.x * pixels_per_thread + j];
		}
		else
		{
			tmp_image_data[j] = 0;
		}
	}
	for(j = 0; j < HISTOGRAM_SIZE; ++j)
	{
		tmp_histogram[j] = 0;
	}

	__syncthreads();

	// Calculate
	if(index < N - 1)
	{
		for(j = 0; j < pixels_per_thread; ++j)
		{
			++tmp_histogram[tmp_image_data[threadIdx.x * pixels_per_thread + j]];
		}
	}
	else
	{
		for(j = 0; j < last_thread_pixels; ++j)
		{
			++tmp_histogram[tmp_image_data[threadIdx.x * pixels_per_thread + j]];
		}
	}

	__syncthreads();

	for(j = 0; j < HISTOGRAM_SIZE; ++j)
	{
		histogram[j] += tmp_histogram[j];
	}
}


int main(void)
{
	int nBlk = 512;
	int nThx = 512;
	int N = nBlk * nThx;

	unsigned int size_image = NBR_OF_ELEMENTS * sizeof(unsigned int);
	unsigned long size_histogram = HISTOGRAM_SIZE * sizeof(unsigned long);

	int pixels_per_thread = NBR_OF_ELEMENTS / (N - 1);
	while(pixels_per_thread < 8)
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

	unsigned long * histogram, * d_histogram, * histogram_result;
	unsigned int * image_data, * d_image_data;

	// Alloc space for device copies od input and output data
	printf("Allocation of space for device copies\n");
	cudaMalloc((void **)&d_image_data, size_image);
	cudaMalloc((void **)&d_histogram, size_histogram);

	// Alloc space for host copies of input, output data and setup input values
	printf("Allocation of space for host copies\n");
	image_data = (unsigned int *)malloc(size_image);
	load_image(image_data);
	histogram = (unsigned long *)malloc(size_histogram);
	//clear_histogram(histogram);
	histogram_result = (unsigned long *)malloc(size_histogram);
	//clear_histogram(histogram_result);

	// Copy data to device
	printf("Copy data to device\n");
	cudaMemcpy(d_image_data, image_data, size_image, cudaMemcpyHostToDevice);
	cudaMemcpy(d_histogram, histogram, size_histogram, cudaMemcpyHostToDevice);

	// Lunch calculations() kernel on GPU with N threads
	printf("Calculations\n");
	calculations<<<nBlk, nThx>>>(d_image_data, d_histogram);

	// Copy result back to host
	printf("Copy result to host\n");
	cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);

	// Checkup
	printf("Checkup\n");
/*	int i, j;
	for(i = 0; i < N; ++i)
	{
		if(i < N - 1)
		{
			for(j = 0; j < pixels_per_thread; ++j)
			{
				++histogram_result[image_data[i * pixels_per_thread + j]]; // thread idx * pixels per thread + pixel idx per thread
			}
		}
		else
		{
			for(j = 0; j < last_thread_pixels; ++j)
			{
				++histogram_result[image_data[i * pixels_per_thread + j]]; // thread idx * pixels per thread + pixel idx per thread
			}
		}
	}

	// Checking correctness
	printf("Checking correctness\n");
	if(are_histograms_equal(histogram, histogram_result))
	{
		printf("Histograms are not equal!");
	}
	else*/
	{
		printf("Success!");
		print_histogram(histogram);
	}

	// Cleanup
	printf("Cleanup\n");
	free(histogram);
	free(image_data);
	free(histogram_result);
	cudaFree(d_histogram);
	cudaFree(d_image_data);

	return 0;
}

