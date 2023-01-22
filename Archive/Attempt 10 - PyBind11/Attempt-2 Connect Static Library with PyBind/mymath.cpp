#include <pybind11/pybind11.h>
#include "math_header.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mymath, m) {
    m.def("add", &add);
    m.def("sub", &sub);
}

