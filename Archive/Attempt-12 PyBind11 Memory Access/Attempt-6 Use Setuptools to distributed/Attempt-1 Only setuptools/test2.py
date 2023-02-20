from COO_Matrix import COO_Matrix
import networkx as nx

m1 = COO_Matrix()
m1.init(row=5, col=5, indextype='int64', valuetype='float')

m1.add_value(3, 3, 10.3);
m1.add_value(0, 2, 20.3);
m1.add_value(1, 2, 30.3);
g1 = m1.to_networx_graph()
m1.change()
print("Change done! 909")
m1.print_matrix()
print(nx.to_dict_of_dicts(g1))


m2 = COO_Matrix()
m2.from_networkx_graph(g1)
m2.print_matrix()

