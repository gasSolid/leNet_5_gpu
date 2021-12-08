all:
	nvcc -arch=sm_30 -o main *.cu 

run:
	./main
clean:
	rm main 
