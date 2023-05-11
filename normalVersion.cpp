#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <math.h>
#include <CL/cl.h>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;

#define MAX_SOURCE_SIZE (0x1000)

bool isEdge(int i, int j, int size) {
    return i == 0 || i == size-1 || j == 0 || j == size-1;
}
bool isSecondEdge(int i, int j, int size) {
    return i == 1 || i == size-2 || j == 1 || j == size-2;
}
void initializeMatrix(int size, float* M) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (isEdge(i, j, size)) {
                M[i*size+j] = 0.5;
            } else if (isSecondEdge(i, j, size)) {
                M[i*size+j] = 0.2;
            } else {
                M[i*size+j] = 0.0;
            }
        }
    }
}

void serial_exec(float** M, float** tmp_M, int size)
{
    for (int i = 2; i < size-2; i++) {
        for (int j = 2; j < size-2; j++) {
            double sum = 0;
            sum += (*M)[(i-1) * size + (j-1)];
            sum += (*M)[(i-1) * size + (j+1)];
            sum += (*M)[(i+1) * size + (j-1)];
            sum += (*M)[(i+1) * size + (j+1)];
            double sum2 = 0;
            sum2 += (*M)[(i) * size + (j-2)];
            sum2 += (*M)[(i) * size + (j+2)];
            sum2 += (*M)[(i-2) * size + (j)];
            sum2 += (*M)[(i+2) * size + (j)];
            (*tmp_M)[i*size+j] = (size*0.02) + (sqrt(sum) + sqrt(sum2))*10;
        }
    }
    std::swap(*M, *tmp_M);

}

int main(int argc, char* argv[]){

    float* ser_matrix;
    float* tmp_MS;
    size_t size = 8192;
    int iteration = 100;

    cout << "size is set to " << size; // Display the input value

    cout << "\nIteration is set to " << iteration<<"\n"; // Display the input value

    ser_matrix = (float *)malloc(sizeof(float)*size*size);
    tmp_MS = (float *)malloc(sizeof(float)*size*size);

    initializeMatrix(size, ser_matrix);
    initializeMatrix(size, tmp_MS);
    
    clock_t begin = clock();

    cout << "Running the Serial version with size :\n"<< size<<"\n";
    for (int i = 0; i < iteration; i++) {
        serial_exec(&ser_matrix, &tmp_MS, size);
    }

    clock_t end = clock();

    cout << "Serial version took "<<((double)(end-begin) / CLOCKS_PER_SEC)<<" second(s)"<<"\n";

    // Uncomment the code to save the results in a txt file

    // std::ofstream ofs_1("out_m_normal.txt", std::ofstream::out);
    //     for (int i = 0; i < size * size; i++)
    //     {
    //         ofs_1 << " " << ser_matrix[i];
    //         if((i+1) % size == 0) ofs_1 << endl;
    //     }
    //     ofs_1.close();


    float* p_matrix;
    float* tmp_M;
    
    p_matrix = (float *)malloc(sizeof(float)*size*size);
    tmp_M = (float *)malloc(sizeof(float)*size*size);

    initializeMatrix(size, p_matrix);
    initializeMatrix(size, tmp_M);

    // OpenCL Part:
    {   
        FILE *kernelFile;
        char *kernelSource;
        size_t kernelSize;
        int dim = size * size * sizeof(float);

        cout<<"Running parallel part:\n";
        kernelFile = fopen("openclKernel.cl", "r");

        if (!kernelFile) {
            cout<<"No file named openclKernel.cl was found\n";
            exit(-1);
        }

        kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
        kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
        fclose(kernelFile);

        // Getting platform and device information
        cl_platform_id platformId = NULL;
        cl_device_id deviceID = NULL;
        cl_uint errNumDevices;
        cl_uint errNumPlatforms;
        cl_int err = clGetPlatformIDs(1, &platformId, &errNumPlatforms);
        err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &errNumDevices);

        // Creating context.
        cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL,  &err);

        // Creating command queue
        cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &err);

        // build options optimization
        const char* buildOptions = "-cl-single-precision-constant";

        // Memory buffers for each array
        cl_mem parMbuff = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, dim, p_matrix, &err);
        cl_mem tempMbuff = clCreateBuffer(context, CL_MEM_USE_HOST_PTR, dim, tmp_M, &err);
        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, (const size_t *)&kernelSize, &err);
        err = clBuildProgram(program, 1, &deviceID, buildOptions, NULL, NULL);
        
        clock_t beginp = clock();
        
        size_t workGroupDim[2] = {32,32};
        size_t globalWorkSpace[2] = {size,size};

        cl_kernel kernel = clCreateKernel(program, "myKernel", &err);

        for(int t=0;t<iteration;t++){
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&parMbuff);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&tempMbuff);
            err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSpace, workGroupDim, 0, NULL, NULL);
            swap(parMbuff,tempMbuff);
        }

        if (err != CL_SUCCESS)
        {
            switch (err)
            {
                case CL_INVALID_PROGRAM:
                    std::cerr << "Error: CL_INVALID_PROGRAM" << std::endl;
                    break;
                case CL_INVALID_VALUE:
                    std::cerr << "Error: CL_INVALID_VALUE" << std::endl;
                    break;
                default:
                    std::cerr << "Error: Unknown error code" << std::endl;
                    break;
            }
        }else{
            cout<<"There were no errors\n";
        }

        err = clEnqueueReadBuffer(commandQueue, parMbuff, CL_TRUE, 0, dim, p_matrix, 0, NULL, NULL);
        clock_t endp = clock();

        cout << "Parallel part took "<<((double)(endp-beginp) / CLOCKS_PER_SEC)<<" second(s)"<<"\n";
        cout << "Speed-up: "<<ceil(((double)(end-begin) / CLOCKS_PER_SEC)/((double)(endp-beginp) / CLOCKS_PER_SEC)) << " Times\n";

        float eps = 1e-5;
        int not_matched = 0;
        for(int i=0;i<size;i++){
            for(int j=0;j<size;j++){
                if(fabs(p_matrix[i+j*size] - ser_matrix[i+j*size])/ser_matrix[i+j*size] > eps){
                    not_matched++;
                }
            }
        }


        if(not_matched > 0){
            cout<<not_matched<<" of cells do not exactly match\n";
        }else{
            cout<<"The results from both version match\n";
        }

        // Uncomment the code to save output in a txt file

        // std::ofstream ofs_2("out_p_normal.txt", std::ofstream::out);
        // for (int i = 0; i < size; i++)
        // {
        //     for(int j = 0; j < size; j++){
        //         ofs_2 << " " << p_matrix[i+j*size];
        //     }
        // }
        // ofs_2.close();

        // Clean up, release memory.
        err = clFlush(commandQueue);
        err = clFinish(commandQueue);
        err = clReleaseCommandQueue(commandQueue);
        err = clReleaseKernel(kernel);
        err = clReleaseProgram(program);
        err = clReleaseMemObject(parMbuff);
        err = clReleaseMemObject(tempMbuff);
        err = clReleaseContext(context);
    }



    // // The followig code prints out two random sub matrices form both serial and parallel version outputs to see the result
    // // sr_sm and sc_sm are set to find a random int between 2 and size of the matrix
    int sr_sm = 2 + ( std::rand() % ( size - 2 + 1 ) );
    int sc_sm = 2 + ( std::rand() % ( size - 2 + 1 ) );

    printf("Arbitrary output from parallel version\n");
    for(int i=sr_sm;i<sr_sm+2;i++){
        for(int j=sc_sm;j<sc_sm+2;j++){
            printf("%f\t\t\t",p_matrix[i+j*size]);
        }
        printf("\n");
    }
    printf("\n");
    printf("Arbitrary output from serial version\n");
    for(int i=sr_sm;i<sr_sm+2;i++){
        for(int j=sc_sm;j<sc_sm+2;j++){
            printf("%f\t\t\t",ser_matrix[i+j*size]);
        }
        printf("\n");
    }

    return 0;
}