#include <iostream>
#include <cstdlib>
#include <functional>
#include <string>
#include <string.h>
using namespace std;

// A simple class with a constuctor and some methods...
template <typename T>
class Graph
{
    public:
        Graph(int n, int isBidirectional)
		{
			N = n;
			this->isBidirectional = isBidirectional;
			adjMat = (T **) calloc(n, (sizeof (T *)));
			for (int i = 0; i< n; i++)
				adjMat[i] = (T *) calloc(n, sizeof(T));
			printf("[C++] A blank %dx%d graph created\n", N, N);
		}
		

		void printGraph()
		{
			printf("\n[C++] Graph (%d x %d):\n", N, N);
			for (int i = 0; i<N; i++){
				for (int j = 0; j<N; j++)
					cout << adjMat[i][j] << "\t";
				printf("\n");
			}
		}

		void printGraph(char *abc)
		{
			//printf("\nString here %s \n",abc);
			printf("\nString here %s \n",abc);
			printf("\n[C++] Graph (%d x %d):\n", N, N);
			for (int i = 0; i<N; i++){
				for (int j = 0; j<N; j++)
					cout << adjMat[i][j] << "\t";
				printf("\n");
			}
		}

        void setEdge(int x, int y, T val)
		{
			if ((x > N -1) || (y > N - 1))
				{
				    printf("[C++] Invalid index!\n");
				    return;
				}
				adjMat[x][y] = val;
				if (isBidirectional)
				    adjMat[y][x] = val;
				printf("[C++] Edge added between %d and %d\n", x, y);
		}
		
		void performOp(const std::function<T(T)> &op){
			printf("[C++] Performing op...");
			for (int i = 0; i<N; i++){
				for (int j = 0; j<N; j++)
				    adjMat[i][j] = op(adjMat[i][j]);
			}

			printf("Done. After Op, the graph looks like the following:");
			this->printGraph();
		}

		
        int countEdge(){
			int count = 0;
			for (int i = 0; i<N; i++){
				for (int j = 0; j<N; j++)
				    if (adjMat[i][j] != 0) count++;
			}
			if (isBidirectional)
				return count / 2;
			return count;
		}
        ~Graph()
		{
			for (int i = 0; i< N; i++)
				free(adjMat[i]);
			free(adjMat);
			adjMat = NULL;
			N = 0;
			printf("[C++] Graph Destroyed\n");
		}
		
    private:
        T ** adjMat;
        int N;
        int isBidirectional;
};

