output: main.o 
	nvcc main.o -o output
main.o: main.cu 
	nvcc -c main.cu
clean:
	rm *.o output