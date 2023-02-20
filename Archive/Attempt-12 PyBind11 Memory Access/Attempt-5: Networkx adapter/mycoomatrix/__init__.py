from .mycoomatrix import *
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx

class COO_Matrix:
	def init(self, row, col, indextype, valuetype):
		self.row = row
		self.col = col
		if indextype == 'int' and valuetype == 'float':
			self.myobj = COO_Matrix_Small()
		if indextype == 'int' and valuetype == 'int':
			self.myobj = COO_Matrix_int()
		if indextype == 'int64' and valuetype == 'float':
			self.myobj = COO_Matrix_large()
	

	def __getattr__(self, attr):
		if attr in self.__dict__:
			return getattr(self, attr)
		return getattr(self.myobj, attr)
		
	def change(self):
		m1_val = np.asarray(memoryview(self.myobj.get_val_ptr()))
		m1_val[0] = 909
	
	def to_networx_graph(self):
		m1_row = np.asarray(memoryview(self.myobj.get_row_ptr()))
		m1_col = np.asarray(memoryview(self.myobj.get_col_ptr()))
		m1_val = np.asarray(memoryview(self.myobj.get_val_ptr()))
		
		mat = coo_matrix((m1_val, (m1_row, m1_col)), shape=(self.row, self.col))
		# print(mat.toarray())
		G = nx.from_scipy_sparse_array(mat, create_using=nx.DiGraph)
		return G
	
	def from_networkx_graph(self, G):
		A = nx.to_scipy_sparse_array(G, format='coo')
		self.row = A.shape[0]
		self.col = A.shape[1]
		self.myobj = COO_Matrix_Small()
		for i, j, v in zip(A.row, A.col, A.data):
			self.myobj.add_value(i, j, v)

		
# Also try: https://www.techiedelight.com/convert-array-vector-cpp/
		
