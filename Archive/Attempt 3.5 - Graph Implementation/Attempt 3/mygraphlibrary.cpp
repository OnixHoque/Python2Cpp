// Ref: https://www.auctoris.co.uk/2017/04/29/calling-c-classes-from-python-with-ctypes/
//------

#include <stdio.h>
#include <stdlib.h>

// extern "C" // required when using C++ compiler

extern "C"
{
    int * Array_new(int n){
        int * ret = (int *) calloc(n, sizeof(int));
        return ret;
    }
    void Array_print(int * arr, int N){

        for (int i = 0; i<N; i++){
            printf("%d\t", arr[i]);
        }
    }

    void Array_set(int * arr, int i, int x){

        arr[i] = x;
    }

    void Array_destroy(int * arr){
        free(arr);
    }
    
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
