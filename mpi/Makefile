TARGET = mpi
COMPILER = gcc
RUN_PROG = mpiexec
SRC_FILES = .\mpi.c .\my_timers.c
LIB_PATH = "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
COMPILE_FLAGS = -fopenmp -trigraphs -Wno-discarded-qualifiers -I$(LIB_PATH)
RUN_FLAGS = -np 4

all: clean build run 

build:
	$(COMPILER) $(SRC_FILES) -o $(TARGET) $(COMPILE_FLAGS)

clean:
	del $(TARGET).exe

run:
	$(RUN_PROG) $(RUN_FLAGS) $(TARGET)
