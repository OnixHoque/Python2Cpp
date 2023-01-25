#include <iostream>
#include <cstdlib>
// A simple class with a constuctor and some methods...
class Graph
{
    public:
        Graph(int, int);
        ~Graph();
        void printGraph();
        void setEdge(int, int);
	void performOp(const std::function<int(int)> &op);
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

void Graph::performOp(const std::function<int(int)> &op){
    printf("[C++] Performing op...");
    for (int i = 0; i<N; i++){
        for (int j = 0; j<N; j++)
            adjMat[i][j] = op(adjMat[i][j]);
    }

    printf("Done. After Op, the graph looks like the following:");
    this->printGraph();
}

Graph::~Graph() {
	Graph::destroy();
}

