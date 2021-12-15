TARGET = omp
GCC = gcc
SRC_FILES = omp.c my_timers.c
COMPILE_FLAGS = -fopenmp -trigraphs -Wno-discarded-qualifiers

all: clean build run 

clean: 
	rm -f omp

build:
	$(GCC) $(SRC_FILES) -o $(TARGET) $(COMPILE_FLAGS)

run: 
	./omp
