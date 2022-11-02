from mygraph import Graph

f=Graph(5)
print ("Number of edges", f.countEdge())

f.printGraph()

f.setEdge(1, 2)
f.setEdge(2, 3)
f.setEdge(1, 3)
f.setEdge(3, 4)
f.setEdge(4, 5)
f.setEdge(5, 6)

f.printGraph()
print ("Number of edges", f.countEdge())


f.destroy()


print ("Number of edges", f.countEdge())
# f.bar()
