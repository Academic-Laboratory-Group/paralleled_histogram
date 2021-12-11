TARGET = omp
GCC = gcc
SRC_FILES = .\omp.c .\my_timers.c
COMPILE_FLAGS = -fopenmp

omp:
	$(GCC) $(SRC_FILES) -o $(TARGET) $(COMPILE_FLAGS) & dir

clean:
	del $(TARGET).exe & dir

run:
	$(TARGET).exe
