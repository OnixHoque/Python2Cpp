
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

void addArraysOnGPU(float* hostArrayA, float* hostArrayB, float* hostResult, int arraySize);

// To check if Python interface is working or not
void test_adder() {
    const int arraySize = 1000; // Adjust the size as needed

    // Host arrays
    float* hostArrayA = new float[arraySize];
    float* hostArrayB = new float[arraySize];
    float* hostResult = new float[arraySize];

    // Initialize host arrays with some data
    for (int i = 0; i < arraySize; ++i) {
        hostArrayA[i] = i;
        hostArrayB[i] = i * 2;
    }

    // Call the function to add arrays on the GPU
    addArraysOnGPU(hostArrayA, hostArrayB, hostResult, arraySize);

    // Print the first few elements of the result array
    for (int i = 0; i < 10; ++i) {
        std::cout << hostResult[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup host memory
    delete[] hostArrayA;
    delete[] hostArrayB;
    delete[] hostResult;

}

void addArraysOnGPU_wrapper(py::array_t<float> A, py::array_t<float> B, py::array_t<float> C, int N){
    float* a = static_cast<float*>(A.mutable_data());
    float* b = static_cast<float*>(B.mutable_data());
    float* c = static_cast<float*>(C.mutable_data());

    addArraysOnGPU(a, b, c, N);

    std::cout << "Array C" << std::endl;

    for (int i = 0; i < 4; ++i) {
        std::cout << c[i] << " ";
    }
}

PYBIND11_MODULE(myadder, m) {
    m.def("test_adder", &test_adder);
    m.def("addArraysOnGPU", &addArraysOnGPU_wrapper);
}