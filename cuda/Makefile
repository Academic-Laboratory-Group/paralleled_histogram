TARGET = cuda
COMPILER = nvcc
SRC_FILES = cuda.cu

all: clean build run

build:
	$(COMPILER) $(SRC_FILES) -o $(TARGET) --compiler-options --trigraphs

clean:
	rm $(TARGET)

run:
	./$(TARGET)
