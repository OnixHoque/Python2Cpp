#include <pybind11/pybind11.h>
#include <vector>
using namespace std;
namespace py = pybind11;

int add(int i, int j) {
	return i + j;
}

extern "C" {
int sub(int i, int j) {
	return i - j;
}
}

uint8_t buffer[] = {
    0, 1, 2, 3,
    4, 5, 6, 7
};

vector<int> v(8, 2);

void set_value()
{
	v[1] = -99;
}

void print_vector()
{
	for (int i = 0; i<8; i++){
	printf("%d ", v[i]);
}
}

PYBIND11_MODULE(example, m) {
    m.def("add", &add);
	m.def("printVector", &print_vector);
	m.def("changeVector", &set_value);


m.def("get_memoryview1d", []() {
    return py::memoryview::from_buffer(
        v.data(),               // buffer pointer
        {8},   // buffer size
	{sizeof(int)}
    );
});

}

