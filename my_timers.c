#include<sys/time.h>
#include<stdio.h>
#include"my_timers.h"

static double timestamp_start, timestamp_end;

void start_time(void)
{

  /* Get current time */
  struct timeval ts;
  gettimeofday(&ts, (struct timezone*)NULL);

  /* Store time-stamp in micro-seconds */
  timestamp_start = (double)ts.tv_sec * 1000000.0 + (double)ts.tv_usec;

}

void stop_time(void)
{

  /* Get current time */
  struct timeval ts;
  gettimeofday(&ts, (struct timezone*)NULL);

  /* Store time-stamp in microseconds */
  timestamp_end = (double)ts.tv_sec * 1000000.0 + (double)ts.tv_usec;

}

double elapsed_time(void)
{
  double time_diff;

  /* Compute difference */
  time_diff = timestamp_end - timestamp_start;
  if (time_diff <= 0.0)
    {
      fprintf(stdout,
              "Warning! The timer is not precise enough.\n");
      return 0.0;
    }

/* Return difference in milliseconds */
  return time_diff / 1000.0;
}

void print_time(char *message)
{
  int ms;

  ms = elapsed_time();
  printf("%s %d ms\n", message, ms);
}


