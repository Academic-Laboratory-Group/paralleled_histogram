TARGET = cuda
COMPILER = nvcc
SRC_FILES = cuda.cu

all: clean build run

build:
	$(COMPILER) $(SRC_FILES) -o $(TARGET)

clean:
	rm $(TARGET)

run:
	./$(TARGET)
