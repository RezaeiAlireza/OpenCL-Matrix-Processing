#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <CL/cl.h>
#include <vector>
#include <fstream>
#include <string.h>
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
    size_t size;
    int iteration;
    bool chckFlag = true;
    cout << "Enter size of the matrix(multiples of 32, min 32 and max 8192):\n";
    while(chckFlag)
    {
        string input;
        getline(cin, input);
        stringstream ss(input);
        ss >> size;
        if(size < 4 || size>8192 || (size % 32) != 0)
        {
            cerr<< "Please enter a valid number\n";
        }
        else chckFlag=false;
    }
    cout << "size is set to " << size;
    cout << "\nNumber of iterations(min is 1 and max is 100): ";
    bool chckIter = true;
    while(chckIter)
    {
        string input;
        getline(cin, input);
        stringstream ss(input);
        ss >> iteration;
        if(iteration < 1 || iteration>100)
        {
            cerr<< "Please enter a valid number\n";
        }
        else chckIter=false;
    }
    cout << "Iteration = " << iteration<<"\n";

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

    cout << "Serial version took "<<((double)(end-begin) / CLOCKS_PER_SEC)<<" second(s) to execute"<<"\n";
    int chckout1;
    cout<<"To save results in a txt press 1:";
    cin>>chckout1;
    if(chckout1 == 1){
    std::ofstream ofs_1("out_m.txt", std::ofstream::out);
        for (int i = 0; i < size * size; i++)
        {
            ofs_1 << " " << ser_matrix[i];
            if((i+1) % size == 0) ofs_1 << endl;
        }
        ofs_1.close();
    }
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
        size_t workGroup;
        bool chcklocal = true;
        while(chcklocal)
        {
            string input;
            getline(cin, input);
            stringstream ss(input);
            ss >> workGroup;
            if(workGroup != 1 && workGroup != 2 && workGroup != 4 && workGroup != 8 && workGroup != 16 && workGroup != 32)
            {
                cerr<< "Set workGroup dimension, Choose one of (1,2,4,8,16,32)\n";
            }
            else chcklocal=false;
        }
        size_t workGroupDim[2] = {workGroup,workGroup};
        size_t globalWorkSpace[2] = {size,size};

        cl_kernel kernel = clCreateKernel(program, "myKernel", &err);

        for(int iter = 0; iter < iteration; iter++){
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

        cout << "Parallel part took "<<((double)(endp-beginp) / CLOCKS_PER_SEC)<<" seconds"<<"\n";
        cout << "Speed-up is: "<<ceil(((double)(end-begin) / CLOCKS_PER_SEC)/((double)(endp-beginp) / CLOCKS_PER_SEC)) << " Times!\n";

        //eps set for this program to 0.01 for desired accuray 
        //The difference between each element of matrices from serial and parallel version will be checked to be less than our eps

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
            cout<<"The results from both versions are same\n";
        }
        int chckout2;
        cout<<"To save results in a txt press 1:";
        cin>>chckout2;
        if(chckout2 == 1){
        std::ofstream ofs_2("out_p.txt", std::ofstream::out);
        for (int i = 0; i < size; i++)
        {
            for(int j = 0; j < size; j++){
                ofs_2 << " " << p_matrix[i+j*size];
            }
        }
        ofs_2.close();
        }
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
    printf("Arbitrary output from serial version\n");
    for(int i=sr_sm;i<sr_sm+2;i++){
        for(int j=sc_sm;j<sc_sm+2;j++){
            printf("%f\t\t\t",ser_matrix[i+j*size]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Arbitrary output from parallel version\n");
    for(int i=sr_sm;i<sr_sm+2;i++){
        for(int j=sc_sm;j<sc_sm+2;j++){
            printf("%f\t\t\t",p_matrix[i+j*size]);
        }
        printf("\n");
    }
    return 0;
}