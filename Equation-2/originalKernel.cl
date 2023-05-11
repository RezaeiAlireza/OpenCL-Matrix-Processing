__kernel void myKernel(__global float *par_matrix, __global float *tmp_M) {
    // Get the global index for the current work item
    int first = get_global_id(0);
    int second = get_global_id(1);

    // Get the global size for the kernel
    int firsts = get_global_size(0);
    int seconds = get_global_size(1);

    // Check if the current work item is within the boundaries of the matrix
    if(first >= 2 && first < firsts-2 && second >= 2 && second < seconds-2) {
        // Compute the index for the current work item
        const int currentIndex = second*seconds+first;
        // Compute the indices for the surrounding work items
        const int topLeftIndex = (second-1) * seconds + (first-1);
        const int topRightIndex = (second-1) * seconds + (first+1);
        const int bottomLeftIndex = (second+1) * seconds + (first-1);
        const int bottomRightIndex = (second+1) * seconds + (first+1);
        const int leftIndex = (second) * seconds + (first-2);
        const int rightIndex = (second) * seconds + (first+2);
        const int topIndex = (second-2) * seconds + (first);
        const int bottomIndex = (second+2) * seconds + (first);

        // Compute the value for the current work item
        tmp_M[currentIndex] = native_sqrt(par_matrix[topLeftIndex] +
                            par_matrix[topRightIndex] +
                            par_matrix[bottomLeftIndex] +
                            par_matrix[bottomRightIndex]) *
                            native_sqrt(par_matrix[leftIndex] +
                            par_matrix[rightIndex] +
                            par_matrix[topIndex] +
                            par_matrix[bottomIndex]);
    }
}
