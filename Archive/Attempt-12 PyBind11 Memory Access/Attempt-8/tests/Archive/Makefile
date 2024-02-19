# # See LICENSE.txt for license details.

# #CXX_FLAGS += -std=c++11 -O3 -Wall
# CXX_FLAGS += -std=c++11 -O3 -Wall -shared -fPIC $(python3 -m pybind11 --includes) COO_pybind.cpp -o COO$(python3-config --extension-suffix)
# PAR_FLAG = -fopenmp

# ifneq (,$(findstring icpc,$(CXX)))
# 	PAR_FLAG = -openmp
# endif

# ifneq (,$(findstring sunCC,$(CXX)))
# 	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
# 	PAR_FLAG = -xopenmp
# endif

# ifneq ($(SERIAL), 1)
# 	CXX_FLAGS += $(PAR_FLAG)
# endif

# TESTS = randmat_test 

# $(info    $(CXX_FLAGS))

# $(info    $(CXX))

# .PHONY: all
# all: $(TESTS)

# % : src/%.cpp 
# 	$(CXX) $(CXX_FLAGS) $< -o $@




# .PHONY: clean
# clean:
# 	rm -f $(TESTS) 


# PYBIND_FLAG = python3 -m pybind11 --includes
# PYTHON_CONFIG_FLAG = python3-config --extension-suffix


CC = c++
CFLAGS = -O3 -Wall -shared -std=c++11 -fopenmp -fPIC

# all: hello

COO: 
	$(CC) $(CFLAGS) $(shell python3 -m pybind11 --includes) pybind11_binding/COO_pybind.cpp -o pybind11_binding/COO$(shell python3-config --extension-suffix)
#$(CC) $(CFLAGS) $(shell python3 -m pybind11 --includes) pybind11_binding/CSC_pybind.cpp -o pybind11_binding/CSC$(shell python3-config --extension-suffix)
	


		

