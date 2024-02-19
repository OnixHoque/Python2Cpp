introduction: this(CSC_adder_io.cpp in tests directory in splib) is a file to test CSC_adder.h(in include folder)
------------

set parameters:(to be done before running)
--------------

STEP_1:

find <int option> at the start of int main in this file and assign option to the function {id}(from CSC_adder.h in include folder, also included in the line below for reference) intended to be used
{
usage: CSC<RIT, VT, CPT> add_vec_of_matrices_{id}<RIT, CIT, VT, CPT, NM>(vector<CSC<RIT, VT, CPT> * > & )
where :
NM is type for number_of_matrices to merge
{id} = 
1 => unordered_map with push_back
2 => unordered_map with push_back and locks
3 => unordered_map with symbolic step
4 => heaps with symbolic step
5 => radix sort with symbolic step
6 => row_size_pvector_maintaining_sum with symbolic step
}

Note: {id} = 3,4,5,6 are supposed to do better than 1,2.
*** Note ***: Testing of {id} = 3,4,5,6 are important before going to 1,2.

STEP_2:

1)to use random matrices generated from COO.h, assign input_method to 0
2)to use rand_matrix_gen.m(which is in tests folder) set input_method as 1(need to use this for now)

STEP_3:

assign the variable, show_time, to true for time of execution information, but diff command(if you assign input_method to 1) will show that 1.out, 1.txt are different where the difference being the time_of_execution, printed at last line in 1.txt.



to run CSC_adder_io.cpp :
-------------------------

if input_method is chosen to be 0 in STEP_2 then,

before running, make sure to set the parameters k, x, y, weighted (available at the start of int main() ) where (x,y,weighted) Generate a weighted ER matrix with 2^x rows and columns and y nonzeros per column and is weighted, depending on boolean variable 'weighted'.

while being in tests folder, type
g++ CSC_adder_io.cpp -fopenmp
./a.out > 1.txt
<have to figure out what to compare the result with>



else if input_method is chosen to be 1 in STEP_2 then,

while being in tests folder, run following commands

g++ CSC_adder_io.cpp -fopenmp
./a.out < 1.in > 1.txt
diff 1.out 1.txt

where diff should not give any difference ideally(except when show_time is assigned true) and

1.in has number_of_matrices as first line and then has the format between {.} for 1st matrix, while next (number_of_matrices-1) matrices follow same pattern

{<no_of_rows> <no_of_cols> <nnz>
<colptr_array>(space between each number)
<row_ids array>(space between each number)
<nz_vals array>(space between each number)
}// each line meaning a new line and no double new lines between any two matrices

1.out has the same format as above between {.} while having a new line at the end after nz_vals array

Note: 1.in, 1.out formats are taken care, if you use rand_matrix_gen.m(in tests directory)


to run rand_matrix_gen.m in tests directory:
--------------------------------------------

set the values of => rows, columns, fraction(close to fraction*rows*columns non-zeros per matrix before addition), k(number of matrices), is_double(should the elements be of type double?), infile_name, outfile_name, left, right (elements in matrix before addition will be distributed between left and right).

And then run.

