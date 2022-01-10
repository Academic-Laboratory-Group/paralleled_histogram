#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>

#include "my_timers.h"

//#define ATOMIC

//#define TEST
#ifdef TEST

#include "../assets/test_image.h" // 2 048 pixels
#define NBR_OF_ELEMENTS 2048	  // width * height in 64x32 test image.

#else

#include "../assets/image.h"	// 2 073 600 pixels
#define NBR_OF_ELEMENTS 2073600 // width * height in FullHD. Change for bigger images!

#endif

#define HISTOGRAM_SIZE (255 * 3 + 1) // max color value

void load_image(unsigned int *image_data)
{
	char pixel[3];
	char *data = header_data;

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

	for (unsigned long i = 0L; i < HISTOGRAM_SIZE; ++i)
	{
		if (0L != histogram[i])
		{
			printf("v: %lu, a: %lu\n", i, histogram[i]);
		}
	}

	printf("\n");
}

void clear_histogram(unsigned long *array)
{
	for (unsigned long i = 0L; i < HISTOGRAM_SIZE; ++i)
	{
		array[i] = 0;
	}
}

int main()
{
	unsigned long *histogram;
	unsigned int *image_data;
	
	image_data = (unsigned int *)malloc(sizeof(unsigned int) * NBR_OF_ELEMENTS);
	load_image(image_data);

	histogram = (unsigned long *)malloc(sizeof(unsigned long) * HISTOGRAM_SIZE);
	clear_histogram(histogram);

	omp_set_num_threads(4);

	start_time();
#ifdef ATOMIC
	#pragma omp parallel for shared(image_data, histogram)
        for (int i = 0; i < NBR_OF_ELEMENTS; ++i)
        {
                #pragma omp atomic
                ++histogram[image_data[i]];
        }
#else
	#pragma omp parallel shared(image_data, histogram)
	{
		unsigned long *sub_histogram = (unsigned long *)malloc(sizeof(unsigned long) * HISTOGRAM_SIZE);
		for (int i = 0; i < HISTOGRAM_SIZE; ++i)
		{
			sub_histogram[i] = 0;
		}

		#pragma omp for nowait
		for (int i = 0; i < NBR_OF_ELEMENTS; ++i)
		{
			++sub_histogram[image_data[i]];
		}

		#pragma omp critical
		{
			for (int i = 0; i < HISTOGRAM_SIZE; ++i)
			{
				histogram[i] += sub_histogram[i];
			}
		}
		free(sub_histogram);
	}
#endif
	stop_time();
	print_histogram(histogram);
	print_time("Elapsed:");
	free(histogram);
	free(image_data);
}
