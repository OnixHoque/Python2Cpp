#include <iostream>
#include <vector>
#include <string>
#include <cmath>


#include "../../include/CSC.h"
#include "../../include/COO.h"
#include "../../include/GAP/pvector.h"
#include "../../include/GAP/timer.h"
#include "../../include/CSC_adder.h"
#include "../../include/utils.h"
//#include "pch.h"
//#include <gtest/gtest.h>




using namespace std::chrono;

// TEST(FactorialTest, Negative) {
//     // This test is named "Negative", and belongs to the "FactorialTest"
//     // test case.
//     EXPECT_EQ(2,2);
        
// }




int main(int argc, char* argv[]){
    std::string filename = std::string(argv[1]);
    COO<uint32_t, uint32_t, float> coo;
    coo.ReadMM(filename);
    CSC<uint32_t, float, uint32_t> csc(coo);
    CSC<uint32_t, float, uint32_t> csc1(coo);


    size_t n=csc.get_ncols();
    std::vector<int> column_vector(n, 2);
    pvector<float> column_vector_1(n);
    pvector<float> column_reduce_vector(n);
    
    for(int j = 0; j < n; j++)
    {
		column_vector_1[j]=2;
	}
    
    // std::cout<<"Information for the 2 matrices.."<<std::endl;
    // csc.PrintInfo();
    // std::cout<<"Mat Addition here..."<<std::endl;
    
    // auto start = high_resolution_clock::now();
    // csc.matAddition_2(csc1);
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
 
    // std::cout << "Time taken by function:"<< duration.count() << " microseconds" <<std:: endl;
    std::cout<<"Column Reduce."<<std::endl;
    //csc.column_reduce();
    csc.column_reduce();

    //testing::InitGoogleTest(&argc, argv);
    //return RUN_ALL_TESTS();
    
	return 0;

}
