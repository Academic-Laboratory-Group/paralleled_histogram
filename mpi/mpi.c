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

void clear_histogram(unsigned long * array)
{
	for (unsigned long i = 0L; i < HISTOGRAM_SIZE; ++i)
	{
		array[i] = 0;
	}
}


int main(int argc, char **argv)
{
	int my_rank;
	int size;
	int root = 0;
	int nbr_of_elements = NBR_OF_ELEMENTS;

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
		printf("Root operations\n");
		image_data = (unsigned int *) malloc(sizeof(unsigned int) * nbr_of_elements);
		load_image(image_data); // lets load our image
//		print_image(image_data);
	}

	// MPI_Bcast(&nbr_of_elements, 1, MPI_INT, root, MPI_COMM_WORLD);

	// Subdomain memory allocation and scattering data
	printf("Subdomain\n");

	unsigned int sub_data_length = nbr_of_elements / size;
	sub_image_data = (unsigned int *) malloc(sizeof(unsigned int) * sub_data_length);

	MPI_Scatter(image_data, sub_data_length, MPI_UNSIGNED,
			sub_image_data, sub_data_length, MPI_UNSIGNED,
			root, MPI_COMM_WORLD);

	// Computing
	printf("Computing\n");

	sub_histogram = (unsigned long *) malloc(sizeof(unsigned long) * HISTOGRAM_SIZE);
	histogram = (unsigned long *) malloc(sizeof(unsigned long) * HISTOGRAM_SIZE);
	clear_histogram(sub_histogram);
	clear_histogram(histogram);

	for (unsigned long i = 0; i < sub_data_length; ++i)
	{
		++sub_histogram[sub_image_data[i]];
	}
	printf("Print rank: %d, subhistogram 544: %lu\n", my_rank, sub_histogram[544]);

	MPI_Reduce(sub_histogram, histogram, HISTOGRAM_SIZE, MPI_UNSIGNED_LONG,
			MPI_SUM, root, MPI_COMM_WORLD);

	// Gathering data
	printf("Gathering\n");

	MPI_Gather(sub_image_data, sub_data_length, MPI_UNSIGNED,
			image_data, sub_data_length, MPI_UNSIGNED,
			root, MPI_COMM_WORLD);

	// Print histogram in root
	if (my_rank == root)
	{
		printf("Print histogram 655: %lu\n", histogram[544]);
		print_histogram(histogram);
	}

	// Finalizing
	printf("Finalizing\n");

	MPI_Finalize();
}

