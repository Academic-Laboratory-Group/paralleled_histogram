#include <omp.h>
#include <stdio.h>
#include <math.h>

#include "my_timers.h"
#include "../assets/image.xbm"

#define NBR_OF_ELEMENTS image_width * image_height / 3
// #define NBR_OF_ELEMENTS image_width * image_height
#define histogram_size 0xff + 1


static unsigned long histogram[histogram_size];
unsigned long i; // iterator


void print_histogram()
{
	if(!histogram)
	{
		printf("Histogram has no values!");
		return;
	}	
	
	printf("Histogram values:\n");

	for (i = 0; i < histogram_size; ++i)
	{
		printf("%lu, ", histogram[i]);
	}
}

int main()
{
	omp_set_num_threads(4);

	start_time();

	#pragma omp parallel for schedule(guided)
	for (i = 0; i < NBR_OF_ELEMENTS; ++i)
	{
		++histogram[image_bits[i]];
	}

	stop_time();
	print_time("Elapsed:");
}

