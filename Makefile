omp: omp.c my_timers.c
	gcc omp.c my_timers.c -o omp -fopenmp
