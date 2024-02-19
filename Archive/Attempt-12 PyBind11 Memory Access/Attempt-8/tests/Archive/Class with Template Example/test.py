from mygraph import Graph_int as Graph

def sq(n):
	return n ** 2

def cube(n):
	return n ** 3

def add_two(n):
	return n + 2

f=Graph(5, 1)
f.printGraph()
print ("Number of edges after graph creation = ", f.countEdge())

f.setEdge(1, 2, 5)
f.setEdge(2, 3, 10)
f.setEdge(1, 3, 2)
f.setEdge(3, 4, 7)
f.setEdge(4, 5, 9)
f.setEdge(5, 6, 4)
print("The graph looks the following after adding these edges:")
f.printGraph()
print ("Number of edges", f.countEdge())

print("Here the overloaded function")
f.printGraph("Hello")

print("\nAdding two with all nodes...")
f.performOp(add_two)



print("\nPerforming square on all nodes...")
f.performOp(sq)

print("\nPerforming cube on all nodes...")
f.performOp(cube)

print("\nLambda function is also supported, applying function 2x - 1...")
f.performOp(lambda x: 2*x - 1)

print("\nGraph will be automatically destroyed when Python exits scope...")


