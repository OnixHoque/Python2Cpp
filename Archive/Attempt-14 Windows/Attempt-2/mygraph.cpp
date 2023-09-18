#include <iostream>
#include <cstdlib>
// A simple class with a constuctor and some methods...
class Graph
{
    public:
        Graph(int, int);
        void printGraph();
        void setEdge(int, int);
        void performOp(int (*op)(int));
        int countEdge();
        void destroy();
    private:
        int ** adjMat;
        int N;
        int isBidirectional;
};

Graph::Graph(int n, int isBidirectional)
{
    N = n;
    this->isBidirectional = isBidirectional;
    adjMat = (int **) calloc(n, (sizeof (int *)));
    for (int i = 0; i< n; i++)
        adjMat[i] = (int *) calloc(n, sizeof(int));
    printf("[C++] A blank %dx%d graph created\n", N, N);
}

void Graph::printGraph()
{
    printf("\n[C++] Graph (%d x %d):\n", N, N);
    for (int i = 0; i<N; i++){
        for (int j = 0; j<N; j++)
            printf("%d\t", adjMat[i][j]);
        printf("\n");
    }
}
void Graph::setEdge(int x, int y)
{
    if ((x > N -1) || (y > N - 1))
        {
            printf("[C++] Invalid index!\n");
            return;
        }
        adjMat[x][y] = 1;
        if (isBidirectional)
            adjMat[y][x] = 1;
        printf("[C++] Edge added between %d and %d\n", x, y);
}

int Graph::countEdge(){
    int count = 0;
    for (int i = 0; i<N; i++){
        for (int j = 0; j<N; j++)
            if (adjMat[i][j]) count++;
    }
    if (isBidirectional)
        return count / 2;
    return count;
}

void Graph::destroy()
{
    for (int i = 0; i< N; i++)
        free(adjMat[i]);
    free(adjMat);
    adjMat = NULL;
    N = 0;
    printf("[C++] Graph Destroyed\n");
}

void Graph::performOp(int (*op)(int)){
    printf("[C++] Performing op...");
    // int x = op(10);
    // int x = -1;
    // printf("sample=%d:\n", x);
    for (int i = 0; i<N; i++){
        for (int j = 0; j<N; j++)
            adjMat[i][j] = op(adjMat[i][j]);
    }

    printf("Done. After Op, the graph looks like the following:");
    this->printGraph();
}

// Define C functions for the C++ class - as ctypes can only talk to C...
// extern "C"
// {
//     Graph* Graph_new(int n) {return new Graph(n, 1);}
//     void Graph_print(Graph* foo) {foo->printGraph();}
//     void Graph_setEdge(Graph* foo, int i, int j) {foo->setEdge(i, j);}
//     int Graph_countEdge(Graph* foo) {return foo->countEdge();}
//     void Graph_performOp(Graph * foo, int (*op)(int)) { foo->performOp(op); }
//     void Graph_destroy(Graph* foo) {foo->destroy();}

// }
