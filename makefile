all: normalVersion interactive originalViter50 originalVdouble

normalVersion: normalVersion.cpp
	g++ -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -o normalVersion normalVersion.cpp -lOpenCL

interactive: interactiveVersion.cpp
		g++ -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -o interactive interactiveVersion.cpp -lOpenCL 

originalViter50: Equation-2/originalViter50.cpp
		g++ -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -o originalViter50 Equation-2/originalViter50.cpp -lOpenCL 

originalVdouble: Equation-2/originalVdouble.cpp
		g++ -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -o originalVdouble Equation-2/originalVdouble.cpp -lOpenCL

clean:
	rm normalVersion
	rm interactive
	rm originalViter50
	rm originalVdouble