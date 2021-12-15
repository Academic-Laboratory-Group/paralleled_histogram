#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <string.h>


//#define TEST
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


static unsigned long histogram[HISTOGRAM_SIZE];
static unsigned int mono_image[NBR_OF_ELEMENTS];


int main(int argc, char **argv)
{
	int my_rank;
	int size;
	int root = 0;

	unsigned long * histogram;
	unsigned long * sub_histogram;
	unsigned int * image_data;
	unsigned int * sub_image_data;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Root operations
	if (my_rank == root)
	{
		image_data = (unsigned int *) malloc(sizeof(unsigned int) * NBR_OF_ELEMENTS);
		load_image(image_data); // lets load our image
	}

	// Subdomain memory allocation and scattering data
	int sub_data_length = NBR_OF_ELEMENTS / size;
	sub_image_data = (unsigned int *) malloc(sizeof(unsigned int) * sub_data_length);

	MPI_Scatter(image_data, sub_data_length, MPI_UNSIGNED,
			sub_image_data, sub_data_length, MPI_UNSIGNED,
			root, MPI_COMM_WORLD);

	// Computing
	sub_histogram = (unsigned long *) malloc(sizeof(unsigned long) * HISTOGRAM_SIZE);
	histogram = (unsigned long *) malloc(sizeof(unsigned long) * HISTOGRAM_SIZE);

	for (unsigned long i = 0; i < NBR_OF_ELEMENTS; ++i)
	{
		++sub_histogram[sub_image_data[i]];
	}

	MPI_Reduce(&sub_histogram, &histogram, HISTOGRAM_SIZE, MPI_UNSIGNED_LONG,
			MPI_SUM, root, MPI_COMM_WORLD);

	// Gathering data
	MPI_Gather(sub_image_data, sub_data_length, MPI_UNSIGNED,
			image_data, sub_data_length, MPI_UNSIGNED,
			root, MPI_COMM_WORLD);

	// Print histogram in root
	if (my_rank == root)
	{
		print_histogram(histogram);
	}

	// Cleaning in memory
//	free(histogram);
//	free(sub_histogram);
//	free(image_data);
//	free(sub_image_data);

	// Finalizing
	MPI_Finalize();
}

