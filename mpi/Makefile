TARGET = mpi
COMPILER = mpicc
RUN_PROG = mpiexec
SRC_FILES = mpi.c my_timers.c
COMPILE_FLAGS = -fopenmp -trigraphs -Wno-discarded-qualifiers
RUN_FLAGS = -np 4

all: clean build run

build:
	$(COMPILER) $(SRC_FILES) -o $(TARGET) $(COMPILE_FLAGS)

clean:
	rm $(TARGET)

run:
	$(RUN_PROG) $(RUN_FLAGS) $(TARGET)
