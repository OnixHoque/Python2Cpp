import gc
from mycoomatrix import COO_Matrix_Small

m1 = COO_Matrix_Small()
m1.add_value(0, 0, 10.5);
m1.add_value(3, 4, 3.1416);
m1.print_matrix()
print(); print()

m1_row = memoryview(m1.get_row_ptr())
m1_col = memoryview(m1.get_col_ptr())
m1_val = memoryview(m1.get_val_ptr())

print("Printing arrays from Python:")
print("Row: ", m1_row[0], m1_row[1])
print("Col: ", m1_col[0], m1_col[1])
print("Val: ", m1_val[0], m1_val[1])
print(); print()

print("Setting 2nd value of row as 100 in Python...")
m1_row[1] = 100;
m1.print_matrix()
print(); print()

print("Adding another value in C++...")
m1.add_value(3, 3, 10.3);
m1.print_matrix()
print();
## Has to get the pointers again after modification...
m1_row = memoryview(m1.get_row_ptr())
m1_col = memoryview(m1.get_col_ptr())
m1_val = memoryview(m1.get_val_ptr())
print("Printing arrays from Python:")
print("Row: ", m1_row[0], m1_row[1])
print("Col: ", m1_col[0], m1_col[1])
print("Val: ", m1_val[0], m1_val[1])
m1.print_matrix()
print(); print()

# Simulating out of scope...
del m1
gc.collect()
	
print("Trying to print from memoryview...")
print("Row: ", m1_row[0], m1_row[1])
print("Col: ", m1_col[0], m1_col[1])
print("Val: ", m1_val[0], m1_val[1])

