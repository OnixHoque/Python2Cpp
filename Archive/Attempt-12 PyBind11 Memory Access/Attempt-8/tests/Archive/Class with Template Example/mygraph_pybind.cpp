#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "mygraph.cpp"

/*
int main()
{
	Graph<float> g1(5, 1);

	g1.setEdge(1, 2, 2);
	g1.performOp([](float x) { return x / 3.0; });
	g1.printGraph();
	return 0;
}*/


namespace py = pybind11;


// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types

template<typename T>
void define_graph(py::module &m, std::string classname)
{
	py::class_<Graph<T>>(m, classname.c_str())
		.def(py::init<int, int>())
		//.def("printGraph",py::overload_cast<string abc> (&Graph<T>))
		//.def("printGraph",py::overload_cast<> (&Graph<T>))
		//.def("printGraph", static_cast<void (Graph<T>::*)(string&)>(&Graph<T>::printGraph), "Print the graph")
		.def("printGraph", static_cast<void (Graph<T>::*)(char*)>(&Graph<T>::printGraph), "Print the graph")
		.def("printGraph", static_cast<void (Graph<T>::*)()>(&Graph<T>::printGraph), "Print the graph")
		// .def("printGraph", &Graph<T>::printGraph)
		.def("setEdge", &Graph<T>::setEdge)
		.def("countEdge", &Graph<T>::countEdge)
		.def("performOp", [](Graph<T> &a, const std::function<T(T)> &f) { a.performOp(f); });
}

PYBIND11_MODULE(mygraph, m) {
	define_graph<float>(m, "Graph_float");
	define_graph<int>(m, "Graph_int");
	define_graph<double>(m, "Graph_double");
	define_graph<char>(m, "Graph_char");
}

/*
PYBIND11_MODULE(mygraph, m) {
	py::class_<Graph<float>>(m, "Graph")
		.def(py::init<int, int>())
		.def("printGraph", &Graph<float>::printGraph)
		.def("setEdge", &Graph<float>::setEdge)
		.def("countEdge", &Graph<float>::countEdge)
		.def("performOp", [](Graph<float> &a, const std::function<float(float)> &f) { a.performOp(f); });
}*/


