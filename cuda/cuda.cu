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
	char *data = header_data;

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

	for (unsigned long i = 0L; i < HISTOGRAM_SIZE; ++i)
	{
		if (0L != histogram[i])
		{
			printf("v: %lu, a: %lu\n", i, histogram[i]);
		}
	}

	printf("\n");
}

void clear_histogram(unsigned long * array)
{
	for (unsigned long i = 0L; i < HISTOGRAM_SIZE; ++i)
	{
		array[i] = 0;
	}
}

_global_ void calculations(unsigned int * image_data, unsigned long * histogram)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	histogram[index] = 0;

	// TODO: make calculations method
}


int main(void)
{
	int nbr_of_elements = NBR_OF_ELEMENTS;
	int nBlk = 512; // TODO: Apply correct size
	int nThx = 512;
	int N = nBlk * nThx;
	int size_image = NBR_OF_ELEMENTS * sizeof(unsigned int);
	int size_histogram = HISTOGRAM_SIZE * sizeof(unsigned long);


	unsigned long * histogram, d_histogram;
	unsigned int * image_data, d_image_data;

	// Alloc space for device copies od input and output data
	cudaMalloc((void **)&d_image_data, size_image);
	cudaMalloc((void **)&d_histogram, size_histogram);

	// Alloc space for host copies of input, output data and setup input values
	image_data = (unsigned int *)malloc(size_image);
	load_image(image_data);
	histogram = (unsigned long *)malloc(size_histogram);

	// Copy input data to image
	cudaMemcpy(d_image_data, image_data, size_image, cudaMemcpyHostToDevice);

	// Lunch calculations() kernel on GPU with N threads
	calculations<<<nBlk, nThx>>>(d_image_data, d_histogram);

	// Copy result back to host
	cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);

	// Checkup
	// TODO: Make checkup


	// Cleanup
	free(histogram);
	free(image_data);
	cudaFree(d_histogram);
	cudaFree(d_image_data);

	return 0;
}

