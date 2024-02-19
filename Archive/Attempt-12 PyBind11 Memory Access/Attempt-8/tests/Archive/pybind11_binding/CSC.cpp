#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <tuple>
#include <random>
#include <vector>


// #include "../../include/COO.h"
#include "../../include/utils.h"
#include "../../include/GAP/timer.h"
#include "../../include/GAP/util.h"
#include "../../include/GAP/pvector.h"
#include "../../include/GAP/platform_atomics.h"

#include <numeric>



/*
 We heavily use pvector (equivalent to std::vector).
Since pvector uses size_t for indexing, we will also stick to  size_t for indexing. That means, nnz, nrows, ncols are all of size_t.
What should we do for value type for array of offsets such as colptr?
size_t is definitely safe because of the above argument.
However, it may take more space for tall and skinny matrices.
Hence, we use an optional type CPT for such cases. By default, CPT=size_t
*/



// RIT: Row Index Type
// VT: Value Type
// CPT: Column pointer type (use only for very tall and skinny matrices)
template <typename RIT, typename VT=double, typename CPT=size_t>
class CSC
{
public:
	CSC(): nrows_(0), ncols_(0), nnz_(0), isColSorted_(false) {}

	// CSC(RIT nrows, size_t ncols, size_t nnz,bool col_sort_bool, bool isWeighted): nrows_(nrows), ncols_(ncols), nnz_(nnz), isColSorted_(col_sort_bool), isWeighted_(isWeighted) 
    // {
	// 	rowIds_.resize(nnz); 
    //     colPtr_.resize(ncols+1); 
    //     nzVals_.resize(nnz);
    // }  // added by abhishek

	template <typename CIT>
	CSC(COO<RIT, CIT, VT> & cooMat);
	
	template <typename AddOp>
	void MergeDuplicateSort(AddOp binop);
	void PrintInfo();

	const pvector<CPT>* get_colPtr(); // added by abhishek
	const pvector<RIT>* get_rowIds(); 
	const pvector<VT>* get_nzVals();
    
    const CPT get_colPtr(size_t idx);

	// CSC<RIT, VT, CPT>(CSC<RIT, VT, CPT> &&other): nrows_(other.nrows_),ncols_(other.ncols_),nnz_(other.nnz_),isWeighted_(other.isWeighted_),isColSorted_(other.isColSorted_)   // added by abhishek
	// {
	// 	rowIds_.resize(nnz_); colPtr_.resize(ncols_+1); nzVals_.resize(nnz_);
	// 	colPtr_ = std::move(other.colPtr_);
	// 	rowIds_ = std::move(other.rowIds_);
	// 	nzVals_ = std::move(other.nzVals_);
	// }

	// CSC<RIT, VT, CPT>& operator= (CSC<RIT, VT, CPT> && other){ // added by abhishek
	// 	nrows_ = other.nrows_;
	// 	ncols_ = other.ncols_;
	// 	nnz_ = other.nnz_;
	// 	isWeighted_ = other.isWeighted_;
	// 	isColSorted_ = other.isColSorted_;
	// 	rowIds_.resize(nnz_); colPtr_.resize(ncols_+1); nzVals_.resize(nnz_);
	// 	colPtr_ = std::move(other.colPtr_);
	// 	rowIds_ = std::move(other.rowIds_);
	// 	nzVals_ = std::move(other.nzVals_);
	// 	return *this;
	// }
    
    bool operator== (const CSC<RIT, VT, CPT> & other);


	size_t get_ncols(); // added by abhishek
	size_t get_nrows();
	size_t get_nnz() ;

	void nz_rows_pvector(pvector<RIT>* row_pointer) { rowIds_ = std::move(*(row_pointer));} // added by abhishek
	void cols_pvector(pvector<CPT>* column_pointer) {colPtr_ = std::move(*(column_pointer));}
	void nz_vals_pvector(pvector<VT>* value_pointer) {nzVals_ = std::move(*(value_pointer));}

	void count_sort(pvector<std::pair<RIT, VT> >& all_elements, size_t expon); // added by abhishek
	void sort_inside_column();

	void print_all(); // added by abhishek
    
	
	void ewiseApply(VT scalar);
	//void ewiseApply1(VT scalar);


	// template <typename T>
	// void dimApply(std::vector<T> mul_vector);

	template <typename T>
	void dimApply(pvector<T> &mul_vector);

	//template <typename T>
	void column_reduce();

	template <typename T>
	pvector<T> column_reduce_1();

	
	void matAddition(CSC &b);

	void matAddition_1(CSC &b);

	void matAddition_2(CSC &b);

	void matAddition_3(CSC &b);


    void column_split(std::vector< CSC<RIT, VT, CPT>* > &vec, int nsplit);
private:
	size_t nrows_;
	size_t ncols_;
	size_t nnz_;
 
	pvector<CPT> colPtr_;
	pvector<RIT> rowIds_;
	pvector<VT> nzVals_;
	bool isWeighted_;
	bool isColSorted_;
};

template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::ewiseApply(VT scalar)
{
	std::cout<<"Scalar"<<scalar<<std::endl;
	std::cout<<"\n"<<std::endl;
	std::cout<<"Rows, columns and non zero values"<<std::endl;
	std::cout<< nrows_<<" "<<ncols_<<" "<<nnz_<<std::endl;


	for(size_t i = 0; i < nzVals_.size(); i++){
		nzVals_[i]=nzVals_[i]*scalar;
		
	}

	std::cout<<"Nonzero values"<<std::endl;
	for(size_t i = 0; i < nzVals_.size(); i++){
		std::cout<<nzVals_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}

}


template <typename RIT, typename VT, typename CPT>
template<typename T>
void CSC<RIT, VT, CPT>::dimApply(pvector<T> &mul_vector)
{
	for(size_t i = 0; i < colPtr_.size(); i++)
	{
		
		for(size_t j=colPtr_[i];j<colPtr_[i+1];j++)
		{
			nzVals_[j]=nzVals_[j]*mul_vector[i];
		}
	}
	std::cout<<"Nonzero values"<<std::endl;
	for(size_t i = 0; i < nzVals_.size(); i++){
		std::cout<<nzVals_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	
}






template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::column_reduce()
{
	
	size_t n=get_ncols();
	//std::cout<<"n:"<<colPtr_.size()<<std::endl;
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

template <typename RIT, typename VT, typename CPT>
template<typename T>
pvector<T> CSC<RIT, VT, CPT>::column_reduce_1()
{
	size_t n=get_ncols();
	pvector<T> result_vector(n+1);
	for(size_t i = 0; i < colPtr_.size(); i++)
	{
		result_vector[i]=0;
	}
	for(size_t i = 0; i < colPtr_.size(); i++)
	{
		
		for(size_t j=colPtr_[i];j<colPtr_[i+1];j++)
		{
			result_vector[i]=result_vector[i]+nzVals_[j];
		}
	}
	std::cout<<"Final Result"<<std::endl;
	
	return result_vector;
}


template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::matAddition_2(CSC &b)
{
	pvector<CPT> c_colPtr_;
	
	c_colPtr_.resize(ncols_+1);
	for (size_t index_for_initialization = 0; index_for_initialization < ncols_+1; index_for_initialization++)
	{
		c_colPtr_[index_for_initialization]=0;
	}
	size_t i,j,k,m;
	for(i = 0; i < ncols_; i++)
	{
		for(j=colPtr_[i],k=b.colPtr_[i];j<colPtr_[i+1] && k<b.colPtr_[i+1];)
		{
			if(rowIds_[j]==b.rowIds_[k])
			{
				c_colPtr_[i+1]++;
				j++;
				k++;
			}
			else if(rowIds_[j]<b.rowIds_[k])
			{
				c_colPtr_[i+1]++;
				j++;
			}
			else
			{
				c_colPtr_[i+1]++;
				k++;
			}
		}
	}
	while (j<colPtr_[i+1]) 
	{
    	c_colPtr_[i+1]++;
		j++;
  	}
	while (k<colPtr_[i+1]) 
	{
    	c_colPtr_[i+1]++;
		k++;
  	}
	// std::cout<<"Before"<<std::endl;
	// for (size_t index_prefix_sum = 0; index_prefix_sum < c_colPtr_.size(); index_prefix_sum++)
	// {
	// 	std::cout<<c_colPtr_[index_prefix_sum]<<std::endl;
	// }
	std::cout<<"Colptr vector for c is set here"<<std::endl;
	for (size_t index_prefix_sum = 1; index_prefix_sum < c_colPtr_.size(); index_prefix_sum++)
	{
		c_colPtr_[index_prefix_sum]=c_colPtr_[index_prefix_sum]+c_colPtr_[index_prefix_sum-1];
	}
	// std::cout<<"After"<<std::endl;
	// for (size_t index_prefix_sum = 0; index_prefix_sum < c_colPtr_.size(); index_prefix_sum++)
	// {
	// 	std::cout<<c_colPtr_[index_prefix_sum]<<std::endl;
	// }
	
	
	size_t resizing_value=c_colPtr_[c_colPtr_.size()-1];
	std::cout<<"Number of non zeroes here:"<<std::endl;
	std::cout<<resizing_value<<std::endl;

	pvector<RIT> c_rowIds_(resizing_value);
	pvector<VT> c_nzVals_(resizing_value);
	
	for(i = 0; i < ncols_; i++)
	{
		for(j=colPtr_[i],k=b.colPtr_[i],m=c_colPtr_[i];j<colPtr_[i+1] && k<b.colPtr_[i+1] && m<c_colPtr_[i+1];)
		{
			if(rowIds_[j]==b.rowIds_[k])
			{
				c_nzVals_[m]=nzVals_[j]+b.nzVals_[k];
				c_rowIds_[m]=rowIds_[j];
				j++;
				k++;
				m++;

			}
			else if(rowIds_[j]<b.rowIds_[k])
			{
				c_nzVals_[m]=nzVals_[j];
				c_rowIds_[m]=rowIds_[j];
				j++;
				m++;

			}
			else
			{
				c_nzVals_[m]=b.nzVals_[k];
				c_rowIds_[m]=b.rowIds_[k];
				k++;
				m++;

			}

		}
	}


	size_t c_nnz= c_nzVals_.size();
	CSC c(nrows_, ncols_,c_nnz,false,false);

	for (size_t i=0; i<c_colPtr_.size(); i++)
	{
		c.colPtr_.push_back(c_colPtr_[i]);
	}
	
	for (size_t i=0; i<c_rowIds_.size(); i++)
	{
		c.rowIds_.push_back(c_rowIds_[i]);
	}

	for (size_t i=0; i<c_nzVals_.size(); i++)
	{
		c.nzVals_.push_back(c_nzVals_[i]);
	}

	std::cout<<"Resultant Final matrix information"<<std::endl;
	// c.PrintInfo();
	c.print_all();
}




template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::matAddition_3(CSC &b)
{
	pvector<CPT> c_colPtr_;
	pvector<RIT> c_rowIds_;
	pvector<VT> c_nzVals_;
	c_colPtr_.resize(ncols_+1);
	for (size_t index_for_initialization = 0; index_for_initialization < ncols_; index_for_initialization++)
	{
		/* code */
		c_colPtr_[index_for_initialization]=0;
	}
	size_t i,j,k,m;
	for(i = 0; i < ncols_; i++)
	{
		for(j=colPtr_[i],k=b.colPtr_[i];j<colPtr_[i+1] && k<b.colPtr_[i+1];)
		{
			if(rowIds_[j]==b.rowIds_[k])
			{
				c_colPtr_[i+1]++;
				j++;
				k++;
			}
			else if(rowIds_[j]<b.rowIds_[k])
			{
				c_colPtr_[i+1]++;
				j++;
			}
			else
			{
				c_colPtr_[i+1]++;
				k++;
			}
		}
	}
	while (j<colPtr_[i+1]) 
	{
    	c_colPtr_[i+1]++;
		j++;
  	}
	while (k<colPtr_[i+1]) 
	{
    	c_colPtr_[i+1]++;
		k++;
  	}
	for (size_t index_prefix_sum = 1; index_prefix_sum < c_colPtr_.size(); index_prefix_sum++)
	{
		c_colPtr_[index_prefix_sum]=c_colPtr_[index_prefix_sum]+c_colPtr_[index_prefix_sum-1];
	}
	
	std::cout<<"HEYYYY"<<std::endl;

	size_t resizing_value=c_colPtr_[c_colPtr_.size()-1];
	//std::cout<<resizing_value<<std::endl;
	c_rowIds_.resize(resizing_value);
	c_nzVals_.resize(resizing_value);

	for(i = 0; i < ncols_; i++)
	{
		for(j=colPtr_[i],k=b.colPtr_[i],m=c_colPtr_[i];j<colPtr_[i+1] && k<b.colPtr_[i+1] && m<c_colPtr_[i+1];)
		{
			if(rowIds_[j]==b.rowIds_[k])
			{
				c_nzVals_[m]=nzVals_[j]+b.nzVals_[k];
				c_rowIds_[m]=rowIds_[j];
				j++;
				k++;
				m++;

			}
			else if(rowIds_[j]<b.rowIds_[k])
			{
				c_nzVals_[m]=nzVals_[j];
				c_rowIds_[m]=rowIds_[j];
				j++;
				m++;

			}
			else
			{
				c_nzVals_[m]=b.nzVals_[k];
				c_rowIds_[m]=b.rowIds_[k];
				k++;
				m++;

			}

		}
	}

	size_t c_nnz= c_nzVals_.size();
	CSC c(nrows_, ncols_,c_nnz,false,false);

	for (size_t i=0; i<c_colPtr_.size(); i++)
	{
		c.colPtr_.push_back(c_colPtr_[i]);
	}
	
	for (size_t i=0; i<c_rowIds_.size(); i++)
	{
		c.rowIds_.push_back(c_rowIds_[i]);
	}

	for (size_t i=0; i<c_nzVals_.size(); i++)
	{
		c.nzVals_.push_back(c_nzVals_[i]);
	}
	c.PrintInfo();
}

template <typename RIT, typename VT, typename CPT>
const pvector<CPT>* CSC<RIT, VT, CPT>::get_colPtr()
{
	return &colPtr_;
}

template <typename RIT, typename VT, typename CPT>
const CPT CSC<RIT, VT, CPT>::get_colPtr(size_t idx)
{
    return colPtr_[idx];
}

template <typename RIT, typename VT, typename CPT>
const pvector<RIT>*  CSC<RIT, VT, CPT>::get_rowIds()
{
	return &rowIds_;
}

template <typename RIT, typename VT, typename CPT>
const pvector<VT>* CSC<RIT, VT, CPT>::get_nzVals()
{
	return &nzVals_;
} 



template <typename RIT, typename VT, typename CPT>
size_t CSC<RIT, VT, CPT>:: get_ncols()
{
	return ncols_;
}

template <typename RIT, typename VT, typename CPT>
size_t CSC<RIT, VT, CPT>:: get_nrows()
{
	return nrows_;
}

template <typename RIT, typename VT, typename CPT>
size_t CSC<RIT, VT, CPT>:: get_nnz()
{
	return nnz_;
}

template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::print_all()
{
	//std::cout << "CSC matrix: " << " Rows= " << nrows_  << " Columns= " << ncols_ << " nnz= " << nnz_ << std::endl<<"column_pointer_array"<<std::endl;
	std::cout<< nrows_<<" "<<ncols_<<" "<<nnz_<<std::endl;
	
	for(size_t i = 0; i < colPtr_.size(); i++){
		std::cout<<colPtr_[i];
		if(i != ncols_){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	//std::cout<<std::endl;
	//std::cout<<std::endl<<"row_correspondents"<<std::endl;
	for(size_t i = 0; i < rowIds_.size(); i++){
		std::cout<<rowIds_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	//std::cout<<std::endl;
	//std::cout<<std::endl<<"nz_values"<<std::endl;
	for(size_t i = 0; i < nzVals_.size(); i++){
		std::cout<<nzVals_[i];
		if(i != nnz_-1){
			std::cout<<" ";
		}else{
			std::cout<<std::endl;
		}
	}
	
	//std::cout<<std::endl;

}



//TODO: need parallel code
template <typename RIT, typename VT, typename CPT>
bool CSC<RIT, VT, CPT>::operator==(const CSC<RIT, VT, CPT> & rhs)
{
    if(nnz_ != rhs.nnz_ || nrows_  != rhs.nrows_ || ncols_ != rhs.ncols_) return false;
    bool same = std::equal(colPtr_.begin(), colPtr_.begin()+ncols_+1, rhs.colPtr_.begin());
    same = same && std::equal(rowIds_.begin(), rowIds_.begin()+nnz_, rhs.rowIds_.begin());
    ErrorTolerantEqual<VT> epsilonequal(EPSILON);
    same = same && std::equal(nzVals_.begin(), nzVals_.begin()+nnz_, rhs.nzVals_.begin(), epsilonequal );
    return same;
}

template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::PrintInfo()
{
	std::cout << "CSC matrix: " << " Rows= " << nrows_  << " Columns= " << ncols_ << " nnz= " << nnz_ << std::endl;
}
// Construct an CSC object from COO
// This will be a widely used function
// Optimize this as much as possible
template <typename RIT, typename VT, typename CPT>
template <typename CIT>
CSC<RIT, VT, CPT>::CSC(COO<RIT, CIT, VT> & cooMat)
{
	Timer t;
	t.Start();
	nrows_ = cooMat.nrows();
	ncols_ = cooMat.ncols();
	nnz_ = cooMat.nnz();
	isWeighted_ = cooMat.isWeighted();
	cooMat.BinByCol(colPtr_, rowIds_, nzVals_);
	MergeDuplicateSort(std::plus<VT>());
	isColSorted_ = true;
	t.Stop();
	//PrintTime("CSC Creation Time", t.Seconds());
}



template <typename RIT, typename VT, typename CPT>
template <typename AddOp>
void CSC<RIT, VT, CPT>::MergeDuplicateSort(AddOp binop)
{
	pvector<RIT> sqNnzPerCol(ncols_);
#pragma omp parallel
	{
		pvector<std::pair<RIT, VT>> tosort;
#pragma omp for
        for(size_t i=0; i<ncols_; i++)
        {
            size_t nnzCol = colPtr_[i+1]-colPtr_[i];
            sqNnzPerCol[i] = 0;
            
            if(nnzCol>0)
            {
                if(tosort.size() < nnzCol) tosort.resize(nnzCol);
                
                for(size_t j=0, k=colPtr_[i]; j<nnzCol; ++j, ++k)
                {
                    tosort[j] = std::make_pair(rowIds_[k], nzVals_[k]);
                }
                
                //TODO: replace with radix or another integer sorting
                sort(tosort.begin(), tosort.begin()+nnzCol);
                
                size_t k = colPtr_[i];
                rowIds_[k] = tosort[0].first;
                nzVals_[k] = tosort[0].second;
                
                // k points to last updated entry
                for(size_t j=1; j<nnzCol; ++j)
                {
                    if(tosort[j].first != rowIds_[k])
                    {
                        rowIds_[++k] = tosort[j].first;
                        nzVals_[k] = tosort[j].second;
                    }
                    else
                    {
                        nzVals_[k] = binop(tosort[j].second, nzVals_[k]);
                    }
                }
                sqNnzPerCol[i] = k-colPtr_[i]+1;
          
            }
        }
    }
    
    
    // now squeze
    // need another set of arrays
    // Think: can we avoid this extra copy with a symbolic step?
    pvector<CPT>sqColPtr;
    ParallelPrefixSum(sqNnzPerCol, sqColPtr);
    nnz_ = sqColPtr[ncols_];
    pvector<RIT> sqRowIds(nnz_);
    pvector<VT> sqNzVals(nnz_);

#pragma omp parallel for
	for(size_t i=0; i<ncols_; i++)
	{
		size_t srcStart = colPtr_[i];
		size_t srcEnd = colPtr_[i] + sqNnzPerCol[i];
		size_t destStart = sqColPtr[i];
		std::copy(rowIds_.begin()+srcStart, rowIds_.begin()+srcEnd, sqRowIds.begin()+destStart);
		std::copy(nzVals_.begin()+srcStart, nzVals_.begin()+srcEnd, sqNzVals.begin()+destStart);
	}
	
	// now replace (just pointer swap)
	colPtr_.swap(sqColPtr);
	rowIds_.swap(sqRowIds);
	nzVals_.swap(sqNzVals);
	
}


template <typename RIT, typename VT, typename CPT> 
void CSC<RIT, VT, CPT>::count_sort(pvector<std::pair<RIT, VT> >& all_elements, size_t expon)
{


	size_t num_of_elements = all_elements.size();

	pvector<std::pair<RIT, VT> > temp_array(num_of_elements);
	RIT count[10] = {0};
	size_t index_for_count;

	for(size_t i = 0; i < num_of_elements; i++){
		index_for_count = ((all_elements[i].first)/expon)%10;
		count[index_for_count]++;
	}

	for(int i = 1; i < 10; i++){
		count[i] += count[i-1];
	}

	for(size_t i = num_of_elements-1; i > 0; i--){
		index_for_count = ((all_elements[i].first)/expon)%10;
		temp_array[count[index_for_count] -1] = all_elements[i];
		count[index_for_count]--;
	}


	index_for_count = ((all_elements[0].first)/expon)%10;
	temp_array[count[index_for_count] -1] = all_elements[0];

	all_elements = std::move(temp_array);

	return;
}




// here rowIds vector is to be sorted piece by piece given that there are no row repititions in a single column or any zero element in nzVals
template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::sort_inside_column()
{
	if(!isColSorted_)
	{
		#pragma omp parallel for
				for(size_t i=0; i<ncols_; i++)
				{
					size_t nnzCol = colPtr_[i+1]-colPtr_[i];
					
					pvector<std::pair<RIT, VT>> tosort(nnzCol);
					RIT max_row = 0;
					if(nnzCol>0)
					{
						//if(tosort.size() < nnzCol) tosort.resize(nnzCol);
						
						for(size_t j=0, k=colPtr_[i]; j<nnzCol; ++j, ++k)
						{
							tosort[j] = std::make_pair(rowIds_[k], nzVals_[k]);
							max_row = std::max(max_row, rowIds_[k]);
						}

						//sort(tosort.begin(), tosort.end());
						for(size_t expon = 1; max_row/expon > 0; expon *= 10){
							count_sort(tosort, expon);
						}
						
						size_t k = colPtr_[i];
						rowIds_[k] = tosort[0].first;
						nzVals_[k] = tosort[0].second;
						
						// k points to last updated entry
						for(size_t j=1; j<nnzCol; ++j)
						{
								rowIds_[++k] = tosort[j].first;
								nzVals_[k] = tosort[j].second;
						}
					}
				}
			

			isColSorted_ = true;
	}
	else{
		return;
	}
}

template <typename RIT, typename VT, typename CPT>
void CSC<RIT, VT, CPT>::column_split(std::vector< CSC<RIT, VT, CPT>* > &vec, int nsplits)
{   
    vec.resize(nsplits);
    int ncolsPerSplit = ncols_ / nsplits;
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
#pragma omp for
        for(int s = 0; s < nsplits; s++){
            CPT colStart = s * ncolsPerSplit;
            CPT colEnd = (s < nsplits-1) ? (s + 1) * ncolsPerSplit: ncols_;
            pvector<CPT> colPtr(colEnd - colStart + 1);
            for(int i = 0; colStart+i < colEnd + 1; i++){
                colPtr[i] = colPtr_[colStart + i] - colPtr_[colStart];
            }
            pvector<RIT> rowIds(rowIds_.begin() + colPtr_[colStart], rowIds_.begin() + colPtr_[colEnd]);
            pvector<VT> nzVals(nzVals_.begin() + colPtr_[colStart], nzVals_.begin() + colPtr_[colEnd]);
            vec[s] = new CSC<RIT, VT, CPT>(nrows_, colEnd - colStart, colPtr_[colEnd] - colPtr_[colStart], false, true);
            vec[s]->cols_pvector(&colPtr);
            vec[s]->nz_rows_pvector(&rowIds);
            vec[s]->nz_vals_pvector(&nzVals);
        }
    }
}


