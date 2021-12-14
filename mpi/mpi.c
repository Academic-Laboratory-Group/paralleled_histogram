#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#include "my_timers.h"

//#define TEST
#ifdef TEST

#include "../assets/test_image.h" // 2 048 pixels
#define NBR_OF_ELEMENTS 2048	  // width * height in 64x32 test image.

#else

#define NBR_OF_ELEMENTS 2073600 // width * height in FullHD. Change for bigger images!
#include "../assets/image.h"	// 2 073 600 pixels

#endif

#define histogram_size 255 * 3 + 1 // max color value

static unsigned long histogram[histogram_size];
static unsigned int mono_image[NBR_OF_ELEMENTS];
static unsigned long i; // iterator

void load_image()
{
	char pixel[3];
	char *data = header_data;

	for (i = 0L; i < NBR_OF_ELEMENTS; ++i)
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
	if (!histogram)
	{
		printf("Histogram has no values!");
		return;
	}

	printf("Histogram values:\n");

	for (i = 0L; i < histogram_size; ++i)
	{
		if (0L != histogram[i])
		{
			printf("v: %lu, a: %lu\n", i, histogram[i]);
		}
	}

	printf("\n");
}

int main(int argc, char **argv)
{
	int p, src, dest, rank;
	int tag = 1;
	char mes[50];

	load_image(); // lets load our image

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	start_time();

	if (rank != 0)
	{
		sprintf(mes, "Hello, this is process %d", rank);
		dest = 0;
		MPI_Send(mes, strlen(mes) + 1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}
	if (rank == 0)
	{
		MPI_Recv(mes, 50, MPI_CHAR, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD,
				 &status);
		printf("%s\n", mes);
	}
	//for (i = 0; i < NBR_OF_ELEMENTS; ++i)
	//{
	//	++histogram[mono_image[i]];
	//}

	stop_time();

	MPI_Finalize();

	//	print_image();
	print_time("Elapsed:");
	print_histogram();
}
