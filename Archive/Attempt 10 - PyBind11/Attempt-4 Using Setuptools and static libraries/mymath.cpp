#include <pybind11/pybind11.h>

extern "C" {
int add(int, int);
int sub(int, int);
}

namespace py = pybind11;

PYBIND11_MODULE(mymath, m) {
    m.def("add", &add);
    m.def("sub", &sub);
}

