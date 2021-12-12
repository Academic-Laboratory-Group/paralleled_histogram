#include <omp.h>
#include <stdio.h>
#include <math.h>

#include "my_timers.h"
//#include "../assets/image.h" // 2 073 600 pixels
#include "../assets/test_image.h" // 2 048 pixels

#define NBR_OF_ELEMENTS width * height
//#define NBR_OF_ELEMENTS 100
#define histogram_size 65535 + 1 // max for unsigned short

static unsigned long histogram[histogram_size];

unsigned long i; // iterator
unsigned short mono_image[NBR_OF_ELEMENTS];

void load_image()
{
	char pixel[3];

	for(i = 0; i < NBR_OF_ELEMENTS; ++i)
	{
		HEADER_PIXEL(header_data, pixel);
		mono_image[i] = (unsigned short)pixel[0] + 
					(unsigned short)pixel[1] + 
					(unsigned short)pixel[2];
	}
}

void print_image()
{
	printf("Image printing:\n");

	for (i = (unsigned long)0; i < NBR_OF_ELEMENTS; ++i)
	{
		printf("%hu, ", mono_image[i]);
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

	for (i = (unsigned long)0; i < histogram_size; ++i)
	{
		if( (unsigned long)0 != histogram[i] )
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
	// print_image();
	print_histogram();
	print_time("Elapsed:");
}

