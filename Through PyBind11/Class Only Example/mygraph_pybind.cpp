#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "mygraph.cpp"

namespace py = pybind11;

PYBIND11_MODULE(mygraph, m) {
	py::class_<Graph>(m, "Graph")
		.def(py::init<int, int>())
		.def("printGraph", &Graph::printGraph)
		.def("setEdge", &Graph::setEdge)
		.def("countEdge", &Graph::countEdge)
		.def("performOp", [](Graph &a, const std::function<int(int)> &f) { a.performOp(f); });
}
