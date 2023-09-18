#include <iostream>

long cuda_spmm_test(int);

int main()
{
    for (int i = 8; i <= 4048; i = i + 8)
    {
        auto t = cuda_spmm_test(i);
        std::cout << i << "\t" << t << "\n";
    }
    
    return 0;
}
