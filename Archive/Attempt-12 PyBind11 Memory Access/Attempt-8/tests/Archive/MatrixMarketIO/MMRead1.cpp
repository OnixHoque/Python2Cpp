#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <tuple>
#include <random>
#include <numeric>
#include "../../include/CSC.h"
#include "../../include/COO.h"
#include "../../include/GAP/pvector.h"
#include "../../include/GAP/timer.h"
#include "../../include/GAP/util.h"
#include "../../include/GAP/platform_atomics.h"
#include "../../include/CSC_adder.h"
#include "../../include/utils.h"
//#include "pch.h"
//#include <gtest/gtest.h>
using namespace std::chrono;

extern "C"
{

    void column_reduce()
    {
	    size_t n=CSC.get_ncols();
	    pvector<float> result_vector(n+1);
	    for(size_t i = 0; i < colPtr_.size()-1; i++)
	    {
		    std::cout<<colPtr_[i]<<std::endl;
		    result_vector[i]=0;
	    }
	    for(size_t i = 0; i < colPtr_.size()-1; i++)
	    {
		    std::cout<<"i:"<<i<<std::endl;
		    for(size_t j=colPtr_[i];j<colPtr_[i+1];j++)
		    {
			    //std::cout<<"j:"<<i<<std::endl;
			    result_vector[i]=result_vector[i]+nzVals_[j];
			
		    }
	    }
	    std::cout<<"Final Result"<<std::endl;
	    for(size_t i = 0; i < colPtr_.size()-1; i++)
	    {
		    std::cout<<result_vector[i]<<std::endl;
	    }

    }
}      


int main(){
    std::cout<<"Enter filename"<<std::endl;
    std::string filename;
    std::cin>>filename;
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
    
    
    std::cout<<"Column Reduce."<<std::endl;
    csc.column_reduce();

   
    
	return 0;

}
