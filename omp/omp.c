#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>

#include "my_timers.h"

//#define TEST
#ifdef TEST

#include "../assets/test_image.h" // 2 048 pixels
#define NBR_OF_ELEMENTS 2048 // width * height in 64x32 test image.

#else

#define NBR_OF_ELEMENTS 2073600 // width * height in FullHD. Change for bigger images! 
#include "../assets/image.h" // 2 073 600 pixels

#endif


#define histogram_size 255 * 3 + 1 // max color value


static unsigned long histogram[histogram_size];
static unsigned int mono_image[NBR_OF_ELEMENTS];
static unsigned long i; // iterator


void load_image()
{
	char pixel[3];
	char * data = header_data;

	for(i = 0L; i < NBR_OF_ELEMENTS; ++i)
	{
		HEADER_PIXEL(data, pixel);
		mono_image[i] = (uint8_t)pixel[0] + (uint8_t)pixel[1] + (uint8_t)pixel[2];
	}
}

void print_image()
{
	printf("Image printing:\n");

	for (i = 0L; i < NBR_OF_ELEMENTS; ++i)
	{
		printf("%u, ", mono_image[i]);
	}

	printf("\n");
}

void print_histogram()
{
	if(!histogram)
	{
		printf("Histogram has no values!");
		return;
	}

	printf("Histogram values:\n");
	
	for (i = 0L; i < histogram_size; ++i)
	{
		if( 0L != histogram[i] )
		{
			printf("v: %lu, a: %lu\n", i, histogram[i]);
		}
	}

	printf("\n");
}

int main()
{
	load_image(); // lets load our image

	omp_set_num_threads(4); 

	start_time();

	#pragma omp parallel for schedule(guided)
	for (i = 0; i < NBR_OF_ELEMENTS; ++i)
	{
		++histogram[mono_image[i]];
	}

	stop_time();
//	print_image();
	print_histogram();
	print_time("Elapsed:");
}

