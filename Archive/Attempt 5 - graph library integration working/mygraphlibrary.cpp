// Ref: https://www.auctoris.co.uk/2017/04/29/calling-c-classes-from-python-with-ctypes/
//------

#include <stdio.h>
#include <stdlib.h>

// extern "C" // required when using C++ compiler

struct Graph
{
    int ** adjMat = NULL;
    int N;
    int isBidirectional;

    Graph(int n, int isBidirectional)
    {
        createGraph(n, isBidirectional);
    }
    void createGraph(int n, int isBidirectional)
    {
        N = n;
        this->isBidirectional = isBidirectional;
        adjMat = (int **) calloc(n, (sizeof (int *)));
        for (int i = 0; i< n; i++)
            adjMat[i] = (int *) calloc(n, sizeof(int));

        // printGraph();

        printf("Blank Graph Created %dx%d. Initalized will all zeros.\n", n, n);
    }
    void printGraph()
    {
        printf("\nGraph (%d x %d):\n", N, N);
        for (int i = 0; i<N; i++){
            for (int j = 0; j<N; j++)
                printf("%d\t", adjMat[i][j]);
            printf("\n");
        }
         
    }
    void setEdge(int x, int y){
        if ((x > N -1) || (y > N - 1))
        {
            printf("Invalid index!\n");
            return;
        }
        adjMat[x][y] = 1;
        if (isBidirectional)
            adjMat[y][x] = 1;
        printf("Edge added between %d and %d\n", x, y);
    }
    int countEdge(){
        int count = 0;
        for (int i = 0; i<N; i++){
            for (int j = 0; j<N; j++)
                if (adjMat[i][j]) count++;
        }
        if (isBidirectional)
            return count / 2;
        return count;
    }
    void destroyGraph()
    {
        for (int i = 0; i< N; i++)
            free(adjMat[i]);
        free(adjMat);
        adjMat = NULL;
        N = 0;
        printf("Graph Destroyed\n");
    }
};

extern "C"
{
    Graph * Graph_new(int a, int b) { 
        Graph * ret = new Graph(a, b);
        // ret->createGraph(a, b);
        // ret->printGraph();
        // printf("Edge Count: %d\n", ret->countEdge());
        return ret;
    }

    void Graph_destroy(Graph * g) { g = (Graph *)g; g-> destroyGraph(); delete (g); }
    void Graph_print(Graph * g) { g = (Graph *)g; g-> printGraph(); }
    int Graph_countEdge(Graph * g) { g = (Graph *)g; return g->countEdge(); }
    void Graph_setEdge(Graph * g, int x, int y) { g = (Graph *)g; g->setEdge(x, y); }

    void Graph_check(Graph * g) { g = (Graph *)g; printf("I'm okay. Graph Size: %d\n", g->N); }


    // Foo* Foo_new(int n) {return new Foo(n);}
    // void Foo_bar(Foo* foo) {foo->bar();}
    // int Foo_foobar(Foo* foo, int n) {return foo->foobar(n);}
}


// int main()
// {
//     Graph * g1 = Graph_new(5, 1);
//     Graph_print(g1);
//     printf("Edge Count: %d\n", Graph_countEdge(g1));
//     Graph_setEdge(g1, 0, 1);
//     Graph_setEdge(g1, 1, 2);
//     Graph_setEdge(g1, 3, 4);
//     printf("Edge Count: %d\n", Graph_countEdge(g1));
//     Graph_destroy(g1);
    

// }

// int main()
// {
//     Graph g1;
//     g1.createGraph(5, 1);
//     g1.printGraph();
//     printf("Edge Count: %d\n", g1.countEdge());
//     g1.setEdge(0, 1);
//     g1.setEdge(1, 2);
//     g1.setEdge(3, 4);
//     printf("Edge Count: %d\n", g1.countEdge());
//     g1.destroyGraph();

// }
