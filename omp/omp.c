#include<omp.h>
#include<stdio.h>
#include<math.h>

#include"my_timers.h"

#define NBR_OF_ELEMENTS 50000

int main()
{
	long i,j;
	float a[NBR_OF_ELEMENTS];
	float p;
	
	omp_set_num_threads(4);
 	
	start_time();
 	
	#pragma omp parallel for
 	for( i=0; i<NBR_OF_ELEMENTS; i++)
	{
 		a[i]=0;
 		for(j=0; j<i; j++)
 			a[i]+=(float)j/(float)i;
 	}
 	
	stop_time();
 	print_time("Elapsed:");
}
