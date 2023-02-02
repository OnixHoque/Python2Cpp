#include <pybind11/pybind11.h>
#include <vector>
#include <iostream>

using namespace std;

template <typename INDEXTYPE, typename VALUETYPE>
class COO_Matrix{
public:	
	vector<INDEXTYPE> row;
	vector<INDEXTYPE> col;	
	vector<VALUETYPE> val;

	void add_value(INDEXTYPE i, INDEXTYPE j, VALUETYPE v){
		row.push_back(i);
		col.push_back(j);
		val.push_back(v);
	}
	
	void print_matrix()
	{
		cout << "[C++] Printing Matrix:\n";
		for (size_t i = 0; i< val.size(); i++)
		{
			cout <<"(" << row[i] << ", " << col[i] << ")" << ": " << val[i] << "\n";
		}
	}
	
	~COO_Matrix(){
		cout << "[C++] CSR Matrix destroyed!\n";
	}
};


namespace py = pybind11;

template<typename T, typename V>
void define_coomatrix(py::module &m, std::string classname)
{
	py::class_<COO_Matrix<T, V>>(m, classname.c_str())
		.def(py::init<>())
		.def("add_value", &COO_Matrix<T, V>::add_value)
		.def("print_matrix", &COO_Matrix<T, V>::print_matrix)
		.def("get_row_ptr", [](COO_Matrix<T, V> &M) {
		    return py::memoryview::from_buffer(
			M.row.data(),               // buffer pointer
			{M.row.size()},   // buffer size
			{sizeof(T)}
		    );
		})
		.def("get_col_ptr", [](COO_Matrix<T, V> &M) {
		    return py::memoryview::from_buffer(
			M.col.data(),               // buffer pointer
			{M.col.size()},   // buffer size
			{sizeof(T)}
		    );
		})
		.def("get_val_ptr", [](COO_Matrix<T, V> &M) {
		    return py::memoryview::from_buffer(
			M.val.data(),               // buffer pointer
			{M.val.size()},   // buffer size
			{sizeof(V)}
		    );
		});

}

PYBIND11_MODULE(mycoomatrix, m) {
	define_coomatrix<int, float>(m, "COO_Matrix_Small");
	define_coomatrix<int, int>(m, "COO_Matrix_int");
	define_coomatrix<long long, float>(m, "COO_Matrix_large");
}

// https://pybind11-jagerman.readthedocs.io/en/stable/advanced.html#return-value-policies


