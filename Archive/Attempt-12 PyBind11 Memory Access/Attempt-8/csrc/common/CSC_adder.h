#ifndef CSC_ADDER_H
#define CSC_ADDER_H

#include "CSC.h" // need to check the relative paths for this section
#include "GAP/pvector.h"
#include "GAP/timer.h"
#include "utils.h"
#include "PBBS/radixSort.h"

#include <vector> // needed while taking input
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <iterator>
#include <assert.h>
#include <omp.h>
#include <queue>
#include <tuple>
#include <fstream>

/*
INDEX:
usage: CSC<RIT, VT, CPT> add_vec_of_matrices_{id}<RIT, CIT, VT, CPT, NM>(vector<CSC<RIT, VT, CPT> * > & )
where :
NM is type for number_of_matrices to merge

{id} = 
1 => unordered_map with push_back
2 => unordered_map with push_back and locks
3 => unordered_map with symbolic step
4 => heaps with symbolic step
5 => radix sort with symbolic step
6 => row_size_pvector_maintaining_sum with symbolic step

Note: ParallelPrefixSum from utils.h is used in here

*/


//..........................................................................//
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>
pvector<RIT> symbolicSpMultiAddHashDynamic(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)
{
    double t0, t1, t2, t3;
    CIT num_of_columns = vec_of_matrices[0]->get_ncols();
    NM number_of_matrices = vec_of_matrices.size();
    
    pvector<RIT> nz_per_column(num_of_columns);
    pvector<RIT> flops_per_column(num_of_columns, 0);

    t0 = omp_get_wtime();
#pragma omp parallel for
    for(CIT i = 0; i < num_of_columns; i++){
        for(int k = 0; k < number_of_matrices; k++){
            const pvector<CPT> *colPtr = vec_of_matrices[k]->get_colPtr();
            flops_per_column[i] += (*colPtr)[i+1] - (*colPtr)[i];
        }
    }
    t1 = omp_get_wtime();

    pvector<int64_t> prefix_sum(num_of_columns+1);
    ParallelPrefixSum(flops_per_column, prefix_sum);
    
    int64_t flops_tot = prefix_sum[num_of_columns];
    int64_t flops_per_thread_expected;
    int64_t flops_per_split_expected;
    
    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;
    CIT nthreads;
    CIT nsplits;
    pvector<CIT> splitters;
    pvector<double> ttimes;
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min( nthreads*4, num_of_columns );
            splitters.resize(nsplits);
            ttimes.resize(nthreads);
            flops_per_thread_expected = flops_tot / nthreads;
            flops_per_split_expected = flops_tot / nsplits;
        }
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * flops_per_split_expected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++) {
            CIT colStart = splitters[s];
            CIT colEnd = (s < nsplits-1) ? splitters[s+1] : num_of_columns;
            for(CIT i = colStart; i < colEnd; i++)
            {
                nz_per_column[i] = 0;

                size_t htSize = minHashTableSize;
                while(htSize < flops_per_column[i]) //htSize is set as 2^n
                {
                    htSize <<= 1;
                }
                if(globalHashVec.size() < htSize)
                    globalHashVec.resize(htSize);
                for(size_t j=0; j < htSize; ++j)
                {
                    globalHashVec[j] = -1;
                }
                
                for(NM k = 0; k < number_of_matrices; k++)
                {
                    const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
                    const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
                    for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++)
                    {
                        
                        RIT key = (*row_ids_i)[j];
                        RIT hash = (key*hashScale) & (htSize-1);
                        while (1) //hash probing
                        {
                            
                            if (globalHashVec[hash] == key) //key is found in hash table
                            {
                                break;
                            }
                            else if (globalHashVec[hash] == -1) //key is not registered yet
                            {
                                globalHashVec[hash] = key;
                                nz_per_column[i] ++;
                                break;
                            }
                            else //key is not found
                            {
                                hash = (hash+1) & (htSize-1);
                            }
                        }
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();
    //printf("[Symbolic Hash] Time for parallel section: %lf\n", t1-t0);
    //printf("[Symbolic Hash] Stats of parallel section timing:\n");
    //getStats<double>(ttimes, true);
    //printf("---\n");
    
    // parallel programming ended
    //for (int i = 0 ; i < nz_per_column.size(); i++){
        //fp << nz_per_column[i] << std::endl;
    //}
    //fp.close();
    return std::move(nz_per_column);
    
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t, typename NM>
pvector<RIT> symbolicSpMultiAddHashStatic(std::vector<CSC<RIT, VT, CPT>* > &vec_of_matrices)
{
    double t0, t1, t2, t3;
    CIT num_of_columns = vec_of_matrices[0]->get_ncols();
    NM number_of_matrices = vec_of_matrices.size();
    
    pvector<RIT> nz_per_column(num_of_columns);
    pvector<RIT> flops_per_column(num_of_columns, 0);

    t0 = omp_get_wtime();
#pragma omp parallel for
    for(CIT i = 0; i < num_of_columns; i++){
        for(int k = 0; k < number_of_matrices; k++){
            const pvector<CPT> *colPtr = vec_of_matrices[k]->get_colPtr();
            flops_per_column[i] += (*colPtr)[i+1] - (*colPtr)[i];
        }
    }
    t1 = omp_get_wtime();

    pvector<int64_t> prefix_sum(num_of_columns+1);
    ParallelPrefixSum(flops_per_column, prefix_sum);
    
    int64_t flops_tot = prefix_sum[num_of_columns];
    
    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;
    CIT nthreads;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for(CIT i = 0; i < num_of_columns; i++)
        {
            nz_per_column[i] = 0;

            size_t htSize = minHashTableSize;
            while(htSize < flops_per_column[i]) //htSize is set as 2^n
            {
                htSize <<= 1;
            }
            if(globalHashVec.size() < htSize)
                globalHashVec.resize(htSize);
            for(size_t j=0; j < htSize; ++j)
            {
                globalHashVec[j] = -1;
            }
            
            for(NM k = 0; k < number_of_matrices; k++)
            {
                const pvector<CPT> *col_ptr_i = vec_of_matrices[k]->get_colPtr();
                const pvector<RIT> *row_ids_i = vec_of_matrices[k]->get_rowIds();
                for(CPT j = (*col_ptr_i)[i]; j < (*col_ptr_i)[i+1]; j++)
                {
                    
                    RIT key = (*row_ids_i)[j];
                    RIT hash = (key*hashScale) & (htSize-1);
                    while (1) //hash probing
                    {
                        
                        if (globalHashVec[hash] == key) //key is found in hash table
                        {
                            break;
                        }
                        else if (globalHashVec[hash] == -1) //key is not registered yet
                        {
                            globalHashVec[hash] = key;
                            nz_per_column[i] ++;
                            break;
                        }
                        else //key is not found
                        {
                            hash = (hash+1) & (htSize-1);
                        }
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
    }
    t1 = omp_get_wtime();
    return std::move(nz_per_column);
    
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHashDynamic(std::vector<CSC<RIT, VT, CPT>* > & matrices, pvector<RIT> & nnzPerCol, bool sorted=true)
{
    double t0, t1, t2, t3;
    int nmatrices = matrices.size();

    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<CPT> prefix_sum(ncols+1);
    ParallelPrefixSum(nnzPerCol, prefix_sum);
    
    pvector<CPT> CcolPtr(prefix_sum.begin(), prefix_sum.end());
    pvector<RIT> CrowIds(prefix_sum[ncols]);
    pvector<VT> CnzVals(prefix_sum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefix_sum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefix_sum[ncols];
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    CPT nnzCPerSplitExpected;
    //auto nnzPerColStats = getStats<RIT>(nnzPerCol);
    CIT nthreads;
    CIT nsplits;

    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;

    pvector<double> colTimes(ncols);

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT>> globalHashVec(minHashTableSize);
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols);
            ttimes.resize(nthreads);
            splitters.resize(nsplits);
            nnzCPerThreadExpected = nnzCTot / nthreads;
            nnzCPerSplitExpected = nnzCTot / nsplits;
        }
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * nnzCPerSplitExpected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic) 
        for(size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = s < nsplits-1 ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                //double tc = omp_get_wtime();
                if(nnzPerCol[i] != 0){
                    //----------- preparing the hash table for this column -------
                    size_t htSize = minHashTableSize;
                    while(htSize < nnzPerCol[i])
                    {
                        htSize <<= 1;
                    }   
                    if(globalHashVec.size() < htSize)
                        globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j)
                    {
                        globalHashVec[j].first = -1;
                    }
                
                    //----------- add this column form all matrices -------
                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    
                        for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                        {
                            RIT key = (*rowIds)[j];
                            RIT hash = (key*hashScale) & (htSize-1);
                            VT curval = (*nzVals)[j];
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash].first == key) //key is found in hash table
                                {
                                    globalHashVec[hash].second += curval;
                                    break;
                                }
                                else if (globalHashVec[hash].first == -1) //key is not registered yet
                                {
                                    globalHashVec[hash].first = key;
                                    globalHashVec[hash].second = curval;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }   
                        }
                        //nnzPerThread[tid] += (*colPtr)[i+1] - (*colPtr)[i]; // Would cause false sharing
                    }
               
                    if(sorted)
                    {
                        size_t index = 0;
                        for (size_t j=0; j < htSize; ++j)
                        {
                            if (globalHashVec[j].first != -1)
                            {
                                globalHashVec[index++] = globalHashVec[j];
                            }
                        }
                    
                        // try radix sort
                        //std::sort(globalHashVec.begin(), globalHashVec.begin() + index, sort_less<IT, NT>);
                        //std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                        integerSort<VT>(globalHashVec.data(), index);
                        
                        //std::vector< std::pair<RIT,VT>> temp1(globalHashVec.begin(), globalHashVec.end());
                        //std::vector< std::pair<RIT,VT>> temp2(globalHashVec.begin(), globalHashVec.end());
                        //std::sort(temp1.begin(), temp1.begin() + index);
                        //integerSort<VT>(temp2.data(), index);

                        //for (size_t j = 0; j < index; j++){
                            //if(temp1[j].first != temp2[j].first) printf("Column %d [nnz %d]: j=%d, %d vs %d\n", i, nnzPerCol[i], j, temp1[j].first, temp2[j].first);
                        //}
                    
                        for (size_t j=0; j < index; ++j)
                        {
                            CrowIds[prefix_sum[i]] = globalHashVec[j].first;
                            CnzVals[prefix_sum[i]] = globalHashVec[j].second;
                            prefix_sum[i] ++;
                        }
                    }
                    else
                    {
                        for (size_t j=0; j < htSize; ++j)
                        {
                            if (globalHashVec[j].first != -1)
                            {
                                CrowIds[prefix_sum[i]] = globalHashVec[j].first;
                                CnzVals[prefix_sum[i]] = globalHashVec[j].second;
                                prefix_sum[i] ++;
                            }
                        }
                    }
                }
                //colTimes[i] = omp_get_wtime() - tc;
            }
        }  // parallel programming ended
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    //printf("[Hash]\tTime for parallel section: %lf\n", t1-t0);
    //printf("[Hash]\tStats for parallel section timing:\n");
    //getStats<double>(ttimes, true);
    //printf("***\n\n");

    Timer clock;
    clock.Start();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    //double denseTime = 0;
    //CPT denseCount = 0;
    //for (CPT i = 0; i < ncols; i++){
        //if(nnzPerCol[i] > densityThreshold){
            //denseTime += colTimes[i];
            //denseCount += 1;
        //}
    //}

    //double sparseTime = 0;
    //CPT sparseCount = 0;
    //for (CPT i = 0; i < ncols; i++){
        //if(nnzPerCol[i] <= densityThreshold){
            //sparseTime += colTimes[i];
            //sparseCount += 1;
        //}
    //}
    
    //std::cout << denseCount << "," << denseTime << "," << sparseCount << "," << sparseTime << ",";

    clock.Stop();
    return std::move(sumMat);
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHashStatic(std::vector<CSC<RIT, VT, CPT>* > & matrices, pvector<RIT> & nnzPerCol, bool sorted=true)
{
    double t0, t1, t2, t3;
    int nmatrices = matrices.size();

    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<CPT> prefix_sum(ncols+1);
    ParallelPrefixSum(nnzPerCol, prefix_sum);
    
    pvector<CPT> CcolPtr(prefix_sum.begin(), prefix_sum.end());
    pvector<RIT> CrowIds(prefix_sum[ncols]);
    pvector<VT> CnzVals(prefix_sum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefix_sum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefix_sum[ncols];

    const RIT minHashTableSize = 16;
    const RIT hashScale = 107;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT>> globalHashVec(minHashTableSize);
#pragma omp for schedule(static) 
        for(CPT i = 0; i < ncols; i++){
            //double tc = omp_get_wtime();
            if(nnzPerCol[i] != 0){
                //----------- preparing the hash table for this column -------
                size_t htSize = minHashTableSize;
                while(htSize < nnzPerCol[i])
                {
                    htSize <<= 1;
                }   
                if(globalHashVec.size() < htSize)
                    globalHashVec.resize(htSize);
                for(size_t j=0; j < htSize; ++j)
                {
                    globalHashVec[j].first = -1;
                }
            
                //----------- add this column form all matrices -------
                for(int k = 0; k < nmatrices; k++)
                {
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                
                    for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                    {
                        RIT key = (*rowIds)[j];
                        RIT hash = (key*hashScale) & (htSize-1);
                        VT curval = (*nzVals)[j];
                        while (1) //hash probing
                        {
                            if (globalHashVec[hash].first == key) //key is found in hash table
                            {
                                globalHashVec[hash].second += curval;
                                break;
                            }
                            else if (globalHashVec[hash].first == -1) //key is not registered yet
                            {
                                globalHashVec[hash].first = key;
                                globalHashVec[hash].second = curval;
                                break;
                            }
                            else //key is not found
                            {
                                hash = (hash+1) & (htSize-1);
                            }
                        }   
                    }
                    //nnzPerThread[tid] += (*colPtr)[i+1] - (*colPtr)[i]; // Would cause false sharing
                }
           
                if(sorted)
                {
                    size_t index = 0;
                    for (size_t j=0; j < htSize; ++j)
                    {
                        if (globalHashVec[j].first != -1)
                        {
                            globalHashVec[index++] = globalHashVec[j];
                        }
                    }
                
                    // try radix sort
                    //std::sort(globalHashVec.begin(), globalHashVec.begin() + index, sort_less<IT, NT>);
                    //std::sort(globalHashVec.begin(), globalHashVec.begin() + index);
                    integerSort<VT>(globalHashVec.data(), index);
                    
                    //std::vector< std::pair<RIT,VT>> temp1(globalHashVec.begin(), globalHashVec.end());
                    //std::vector< std::pair<RIT,VT>> temp2(globalHashVec.begin(), globalHashVec.end());
                    //std::sort(temp1.begin(), temp1.begin() + index);
                    //integerSort<VT>(temp2.data(), index);

                    //for (size_t j = 0; j < index; j++){
                        //if(temp1[j].first != temp2[j].first) printf("Column %d [nnz %d]: j=%d, %d vs %d\n", i, nnzPerCol[i], j, temp1[j].first, temp2[j].first);
                    //}
                
                    for (size_t j=0; j < index; ++j)
                    {
                        CrowIds[prefix_sum[i]] = globalHashVec[j].first;
                        CnzVals[prefix_sum[i]] = globalHashVec[j].second;
                        prefix_sum[i] ++;
                    }
                }
                else
                {
                    for (size_t j=0; j < htSize; ++j)
                    {
                        if (globalHashVec[j].first != -1)
                        {
                            CrowIds[prefix_sum[i]] = globalHashVec[j].first;
                            CnzVals[prefix_sum[i]] = globalHashVec[j].second;
                            prefix_sum[i] ++;
                        }
                    }
                }
            }
            //colTimes[i] = omp_get_wtime() - tc;
        }  // parallel programming ended
        ttime = omp_get_wtime() - ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    //printf("[Hash]\tTime for parallel section: %lf\n", t1-t0);
    //printf("[Hash]\tStats for parallel section timing:\n");
    //getStats<double>(ttimes, true);
    //printf("***\n\n");

    Timer clock;
    clock.Start();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    clock.Stop();
    return std::move(sumMat);
}

/*
 * Sliding hash
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHashSlidingDynamic(std::vector<CSC<RIT, VT, CPT>* > & matrices, const RIT windowSizeSymbolic, const RIT windowSize, bool sorted=true)
{
    double t0, t1, t2, t3, t4, t5;

    t0 = omp_get_wtime();
    const RIT minHashTableSize = 16;
    const RIT maxHashTableSize = windowSize;
    const RIT maxHashTableSizeSymbolic = windowSizeSymbolic;
    const RIT hashScale = 107;
    //const RIT minHashTableSize = 2;
    //const RIT maxHashTableSize = 5;
    //const RIT maxHashTableSizeSymbolic = 5;
    //const RIT hashScale = 3;
    CIT nthreads;
    CIT nsplits;
    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<size_t> flopsPerCol(ncols);
    pvector<size_t> nWindowPerColSymbolic(ncols); 
    pvector<RIT> nWindowPerCol(ncols); 
    pvector<RIT> nnzPerCol(ncols, 0);

    t2 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
        }
#pragma omp for
        for(CPT i = 0; i < ncols; i++){
            flopsPerCol[i] = 0;
            for(int k = 0; k < nmatrices; k++){
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                flopsPerCol[i] += (*colPtr)[i+1] - (*colPtr)[i];
            }
            nWindowPerColSymbolic[i] = (flopsPerCol[i] / maxHashTableSizeSymbolic) + 1;
        }
    }
    t3 = omp_get_wtime();

    pvector<size_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);

    pvector<size_t> prefixSumWindowSymbolic(ncols+1, 0);
    ParallelPrefixSum(nWindowPerColSymbolic, prefixSumWindowSymbolic);
#ifdef BREAKDOWN
    // Print total windows needed for symbolic step
    printf("%d,", prefixSumWindowSymbolic[ncols]);
#endif

    pvector < std::pair<RIT, RIT> > nnzPerWindowSymbolic(prefixSumWindowSymbolic[ncols]);
    
    size_t flopsTot = prefixSumSymbolic[ncols];
    size_t flopsPerThreadExpected;
    size_t flopsPerSplitExpected;

    ttimes.resize(nthreads);
    splitters.resize(nsplits);
    flopsPerSplitExpected = flopsTot / nsplits;
    flopsPerThreadExpected = flopsTot / nthreads;
    
    t1 = omp_get_wtime();
    
    size_t cacheL1 = 32 * 1024;
    size_t elementsToFitL1 = cacheL1 / sizeof( std::pair<RIT,RIT> ); // 32KB L1 cache / 8B element size = 4096 elements needed to fit cache line
    size_t padding = std::max(elementsToFitL1, nmatrices); 
    pvector< std::pair< RIT, RIT > > rowIdsRange(padding * nthreads); // Padding to avoid false sharing

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        size_t tid = omp_get_thread_num();
#pragma omp for
        for(size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSumSymbolic.begin(), prefixSumSymbolic.end(), s * flopsPerSplitExpected) - prefixSumSymbolic.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                nnzPerCol[i] = 0;
                nWindowPerCol[i] = 1;
                size_t nwindows = nWindowPerColSymbolic[i];
                if (nwindows == 1){
                    RIT rowStart = 0;
                    RIT  rowEnd = nrows;
                    size_t wIdx = prefixSumWindowSymbolic[i];

                    nnzPerWindowSymbolic[wIdx].first = 0;
                    nnzPerWindowSymbolic[wIdx].second = 0;

                    size_t flopsWindow = flopsPerCol[i];
                    size_t htSize = minHashTableSize;
                    while(htSize < flopsWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j] = -1;
                    }

                    for(size_t k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                        auto first = rowIds->begin() + (*colPtr)[i];
                        auto last = rowIds->begin() + (*colPtr)[i+1];

                        for(; first<last; first++){
                            RIT key = *first;
                            RIT hash = (key*hashScale) & (htSize-1);
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash] == key) //key is found in hash table
                                {
                                    break;
                                }
                                else if (globalHashVec[hash] == -1) //key is not registered yet
                                {
                                    globalHashVec[hash] = key;
                                    nnzPerCol[i]++;
                                    nnzPerWindowSymbolic[wIdx].second++;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }
                        }
                    }
                }
                else{
                    RIT nrowsPerWindow = nrows / nwindows;
                    RIT runningSum = 0;
                    for(size_t w = 0; w < nwindows; w++){
                        RIT rowStart = w * nrowsPerWindow;
                        RIT rowEnd = (w == nwindows-1) ? nrows : (w+1) * nrowsPerWindow;

                        int64_t wIdx = prefixSumWindowSymbolic[i] + w;

                        nnzPerWindowSymbolic[wIdx].first = rowStart;
                        nnzPerWindowSymbolic[wIdx].second = 0;

                        size_t flopsWindow = 0;

                        for(int k = 0; k < nmatrices; k++){
                            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                            const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                            auto first = rowIds->begin() + (*colPtr)[i];
                            auto last = rowIds->begin() + (*colPtr)[i+1];
                            size_t startIdx, endIdx, midIdx;

                            if(rowStart > 0){
                                startIdx = (*colPtr)[i];
                                endIdx = (*colPtr)[i+1];
                                midIdx = (startIdx + endIdx) / 2;
                                while(startIdx < endIdx){
                                    if((*rowIds)[midIdx] < rowStart) startIdx = midIdx + 1;
                                    else endIdx = midIdx;
                                    midIdx = (startIdx + endIdx) / 2;
                                }
                                first = rowIds->begin() + endIdx;
                                //first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                            }

                            if(rowEnd < nrows){
                                startIdx = (*colPtr)[i];
                                endIdx = (*colPtr)[i+1];
                                midIdx = (startIdx + endIdx) / 2;
                                while(startIdx < endIdx){
                                    if((*rowIds)[midIdx] < rowEnd) startIdx = midIdx + 1;
                                    else endIdx = midIdx;
                                    midIdx = (startIdx + endIdx) / 2;
                                }
                                last = rowIds->begin() + endIdx;
                                //last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                            }
                            rowIdsRange[tid * padding + k].first = first - rowIds->begin();
                            rowIdsRange[tid * padding + k].second = last - rowIds->begin();

                            flopsWindow += last-first;
                        }

                        size_t htSize = minHashTableSize;
                        while(htSize < flopsWindow) //htSize is set as 2^n
                        {
                            htSize <<= 1;
                        }
                        if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                        for(size_t j=0; j < htSize; ++j){
                            globalHashVec[j] = -1;
                        }

                        for(int k = 0; k < nmatrices; k++)
                        {
                            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                            const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                            auto first = rowIds->begin() + rowIdsRange[tid  * padding + k].first;
                            auto last = rowIds->begin() + rowIdsRange[tid * padding + k].second;
                            
                            for(; first<last; first++){
                                RIT key = *first;
                                RIT hash = (key*hashScale) & (htSize-1);
                                while (1) //hash probing
                                {
                                    if (globalHashVec[hash] == key) //key is found in hash table
                                    {
                                        break;
                                    }
                                    else if (globalHashVec[hash] == -1) //key is not registered yet
                                    {
                                        globalHashVec[hash] = key;
                                        nnzPerCol[i]++;
                                        nnzPerWindowSymbolic[wIdx].second++;
                                        break;
                                    }
                                    else //key is not found
                                    {
                                        hash = (hash+1) & (htSize-1);
                                    }
                                }
                            }
                        }
                        if (w == 0){
                            //nWindowPerCol[i] = 1;
                            runningSum = nnzPerWindowSymbolic[wIdx].second;
                        }
                        else{
                            if(runningSum + nnzPerWindowSymbolic[wIdx].second > maxHashTableSize){
                                nWindowPerCol[i]++;
                                runningSum = nnzPerWindowSymbolic[wIdx].second;
                            }
                            else{
                                runningSum = runningSum + nnzPerWindowSymbolic[wIdx].second;
                            }
                        }
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();
#ifdef DEBUG
    printf("[Sliding Hash]\tTime for symbolic: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of time consumed by threads:\n");
    getStats<double>(ttimes, true);
#endif
    //printf("---\n");
    
    pvector<RIT> prefixSumWindow(ncols+1, 0);
    ParallelPrefixSum(nWindowPerCol, prefixSumWindow);
#ifdef BREAKDOWN
    // Print time needed for symbolic step
    printf("%lf,", t1-t0);
    // Print total window needed for symbplic step
    printf("%d,", prefixSumWindow[ncols]);
#endif

    //printf("[Sliding Hash]\tStats of number of windows:\n");
    //getStats<RIT>(nWindowPerCol, true);

    pvector< std::pair<RIT, RIT> > nnzPerWindow(prefixSumWindow[ncols]);
    
    t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for(int t = 0; t < nthreads; t++){
        CPT colStart = splitters[t];
        CPT colEnd = (t < nthreads-1) ? splitters[t+1] : ncols;
        for(CPT i = colStart; i < colEnd; i++){
            int64_t nwindows = nWindowPerColSymbolic[i];
            int64_t wsIdx = prefixSumWindowSymbolic[i];
            int64_t wcIdx = prefixSumWindow[i];
            nnzPerWindow[wcIdx].first = nnzPerWindowSymbolic[wsIdx].first;
            nnzPerWindow[wcIdx].second = nnzPerWindowSymbolic[wsIdx].second;
            for(size_t w=1; w < nwindows; w++){
                wsIdx = prefixSumWindowSymbolic[i] + w;
                if(nnzPerWindow[wcIdx].second + nnzPerWindowSymbolic[wsIdx].second > maxHashTableSize){
                    wcIdx++;
                    nnzPerWindow[wcIdx].first = nnzPerWindowSymbolic[wsIdx].first;
                    nnzPerWindow[wcIdx].second = nnzPerWindowSymbolic[wsIdx].second;
                }
                else{
                    nnzPerWindow[wcIdx].second = nnzPerWindow[wcIdx].second + nnzPerWindowSymbolic[wsIdx].second;
                }
            }
        }
    }
    t1 = omp_get_wtime();

    pvector<CPT> prefixSum(ncols+1, 0);
    ParallelPrefixSum(nnzPerCol, prefixSum);

    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);
    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    CPT nnzCPerThreadExpected = nnzCTot / nthreads;
    CPT nnzCPerSplitExpected = nnzCTot / nsplits;
    
    pvector<double> colTimes(ncols);
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT> > globalHashVec(minHashTableSize);
#pragma omp for
        for(size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSum.begin(), prefixSum.end(), s * nnzCPerSplitExpected) - prefixSum.begin();
        }
#pragma omp for schedule(dynamic)
        for(size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                //double tc = omp_get_wtime();
                RIT nwindows = nWindowPerCol[i];
                if(nwindows > 1){
                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                        rowIdsRange[padding * tid + k].first = (*colPtr)[i];
                        rowIdsRange[padding * tid + k].second = (*colPtr)[i+1];
                    }
                    for (int w = 0; w < nwindows; w++){
                        RIT wIdx = prefixSumWindow[i] + w;
                        RIT rowStart = nnzPerWindow[wIdx].first;
                        RIT rowEnd = (w == nWindowPerCol[i]-1) ? nrows : nnzPerWindow[wIdx+1].first;
                        RIT nnzWindow = nnzPerWindow[wIdx].second;

                        size_t htSize = minHashTableSize;
                        while(htSize < nnzWindow) //htSize is set as 2^n
                        {
                            htSize <<= 1;
                        }
                        if(globalHashVec.size() < htSize)
                            globalHashVec.resize(htSize);
                        for(size_t j=0; j < htSize; ++j)
                        {
                            globalHashVec[j].first = -1;
                        }

                        for(int k = 0; k < nmatrices; k++) {
                            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                            const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                            const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                            while( (rowIdsRange[padding * tid + k].first < rowIdsRange[padding * tid + k].second) && ((*rowIds)[rowIdsRange[padding * tid + k].first] < rowEnd) ){
                                RIT j = rowIdsRange[padding * tid + k].first;
                                RIT key = (*rowIds)[j];
                                RIT hash = (key*hashScale) & (htSize-1);
                                VT curval = (*nzVals)[j];
                                while (1) //hash probing
                                {
                                    if (globalHashVec[hash].first == key) //key is found in hash table
                                    {
                                        globalHashVec[hash].second += curval;
                                        break;
                                    }
                                    else if (globalHashVec[hash].first == -1) //key is not registered yet
                                    {
                                        globalHashVec[hash].first = key;
                                        globalHashVec[hash].second = curval;
                                        break;
                                    }
                                    else //key is not found
                                    {
                                        hash = (hash+1) & (htSize-1);
                                    }
                                }

                                rowIdsRange[padding * tid + k].first++;
                            } 
                        }

                        if(sorted){
                            size_t index = 0;
                            for (size_t j=0; j < htSize; ++j){
                                if (globalHashVec[j].first != -1){
                                    globalHashVec[index++] = globalHashVec[j];
                                }
                            }
                    
                            integerSort<VT>(globalHashVec.data(), index);
                        
                            for (size_t j=0; j < index; ++j){
                                CrowIds[prefixSum[i]] = globalHashVec[j].first;
                                CnzVals[prefixSum[i]] = globalHashVec[j].second;
                                prefixSum[i] ++;
                            }
                        }
                        else{
                            for (size_t j=0; j < htSize; ++j)
                            {
                                if (globalHashVec[j].first != -1)
                                {
                                    CrowIds[prefixSum[i]] = globalHashVec[j].first;
                                    CnzVals[prefixSum[i]] = globalHashVec[j].second;
                                    prefixSum[i] ++;
                                }
                            }
                        }
                    }
                    
                }
                else{
                    RIT wIdx = prefixSumWindow[i];
                    RIT nnzWindow = nnzPerWindow[wIdx].second;

                    size_t htSize = minHashTableSize;
                    while(htSize < nnzWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize)
                        globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j)
                    {
                        globalHashVec[j].first = -1;
                    }
                    
                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                        for( RIT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++){
                            RIT key = (*rowIds)[j];
                            RIT hash = (key*hashScale) & (htSize-1);
                            VT curval = (*nzVals)[j];
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash].first == key) //key is found in hash table
                                {
                                    globalHashVec[hash].second += curval;
                                    break;
                                }
                                else if (globalHashVec[hash].first == -1) //key is not registered yet
                                {
                                    globalHashVec[hash].first = key;
                                    globalHashVec[hash].second = curval;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }
                        } 
                    }
                    if(sorted){
                        size_t index = 0;
                        for (size_t j=0; j < htSize; ++j){
                            if (globalHashVec[j].first != -1){
                                globalHashVec[index++] = globalHashVec[j];
                            }
                        }
                
                        integerSort<VT>(globalHashVec.data(), index);
                    
                        for (size_t j=0; j < index; ++j){
                            CrowIds[prefixSum[i]] = globalHashVec[j].first;
                            CnzVals[prefixSum[i]] = globalHashVec[j].second;
                            prefixSum[i] ++;
                        }
                    }
                    else{
                        for (size_t j=0; j < htSize; ++j)
                        {
                            if (globalHashVec[j].first != -1)
                            {
                                CrowIds[prefixSum[i]] = globalHashVec[j].first;
                                CnzVals[prefixSum[i]] = globalHashVec[j].second;
                                prefixSum[i] ++;
                            }
                        }
                    }
                }
                //colTimes[i] = omp_get_wtime() - tc;
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    //printf("%lf,", t1-t0);

#ifdef DEBUG
    printf("[Sliding Hash]\tTime for computation: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of parallel section timings:\n");
    getStats<double>(ttimes, true);
#endif
    //printf("***\n\n");

#ifdef BREAKDOWN
    // Print time needed for symbolic step
    printf("%lf,", t1-t0);
#endif

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);
    return std::move(sumMat);
}

/*
 * Sliding hash
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHashSlidingStatic(std::vector<CSC<RIT, VT, CPT>* > & matrices, const RIT windowSizeSymbolic, const RIT windowSize, bool sorted=true)
{
    double t0, t1, t2, t3, t4, t5;

    t0 = omp_get_wtime();
    const RIT minHashTableSize = 16;
    const RIT maxHashTableSize = windowSize;
    const RIT maxHashTableSizeSymbolic = windowSizeSymbolic;
    const RIT hashScale = 107;
    //const RIT minHashTableSize = 2;
    //const RIT maxHashTableSize = 5;
    //const RIT maxHashTableSizeSymbolic = 5;
    //const RIT hashScale = 3;
    CIT nthreads;
    CIT nsplits;
    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<size_t> flopsPerCol(ncols);
    pvector<size_t> nWindowPerColSymbolic(ncols); 
    pvector<RIT> nWindowPerCol(ncols); 
    pvector<RIT> nnzPerCol(ncols, 0);

    t2 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
        }
#pragma omp for
        for(CPT i = 0; i < ncols; i++){
            flopsPerCol[i] = 0;
            for(int k = 0; k < nmatrices; k++){
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                flopsPerCol[i] += (*colPtr)[i+1] - (*colPtr)[i];
            }
            nWindowPerColSymbolic[i] = (flopsPerCol[i] / maxHashTableSizeSymbolic) + 1;
        }
    }
    t3 = omp_get_wtime();

    pvector<size_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);

    pvector<size_t> prefixSumWindowSymbolic(ncols+1, 0);
    ParallelPrefixSum(nWindowPerColSymbolic, prefixSumWindowSymbolic);

    pvector < std::pair<RIT, RIT> > nnzPerWindowSymbolic(prefixSumWindowSymbolic[ncols]);
    
    size_t flopsTot = prefixSumSymbolic[ncols];
    size_t flopsPerThreadExpected;
    size_t flopsPerSplitExpected;

    ttimes.resize(nthreads);
    splitters.resize(nsplits);
    flopsPerSplitExpected = flopsTot / nsplits;
    flopsPerThreadExpected = flopsTot / nthreads;
    
    t1 = omp_get_wtime();
    
    size_t cacheL1 = 32 * 1024;
    size_t elementsToFitL1 = cacheL1 / sizeof( std::pair<RIT,RIT> ); // 32KB L1 cache / 8B element size = 4096 elements needed to fit cache line
    size_t padding = std::max(elementsToFitL1, nmatrices); 
    pvector< std::pair< RIT, RIT > > rowIdsRange(padding * nthreads); // Padding to avoid false sharing

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector<RIT> globalHashVec(minHashTableSize);
        size_t tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for(CPT i = 0; i < ncols; i++){
            nnzPerCol[i] = 0;
            nWindowPerCol[i] = 1;
            size_t nwindows = nWindowPerColSymbolic[i];
            if (nwindows == 1){
                RIT rowStart = 0;
                RIT  rowEnd = nrows;
                size_t wIdx = prefixSumWindowSymbolic[i];

                nnzPerWindowSymbolic[wIdx].first = 0;
                nnzPerWindowSymbolic[wIdx].second = 0;

                size_t flopsWindow = flopsPerCol[i];
                size_t htSize = minHashTableSize;
                while(htSize < flopsWindow) //htSize is set as 2^n
                {
                    htSize <<= 1;
                }
                if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                for(size_t j=0; j < htSize; ++j){
                    globalHashVec[j] = -1;
                }

                for(size_t k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                    auto first = rowIds->begin() + (*colPtr)[i];
                    auto last = rowIds->begin() + (*colPtr)[i+1];

                    for(; first<last; first++){
                        RIT key = *first;
                        RIT hash = (key*hashScale) & (htSize-1);
                        while (1) //hash probing
                        {
                            if (globalHashVec[hash] == key) //key is found in hash table
                            {
                                break;
                            }
                            else if (globalHashVec[hash] == -1) //key is not registered yet
                            {
                                globalHashVec[hash] = key;
                                nnzPerCol[i]++;
                                nnzPerWindowSymbolic[wIdx].second++;
                                break;
                            }
                            else //key is not found
                            {
                                hash = (hash+1) & (htSize-1);
                            }
                        }
                    }
                }
            }
            else{
                RIT nrowsPerWindow = nrows / nwindows;
                RIT runningSum = 0;
                for(size_t w = 0; w < nwindows; w++){
                    RIT rowStart = w * nrowsPerWindow;
                    RIT rowEnd = (w == nwindows-1) ? nrows : (w+1) * nrowsPerWindow;

                    int64_t wIdx = prefixSumWindowSymbolic[i] + w;

                    nnzPerWindowSymbolic[wIdx].first = rowStart;
                    nnzPerWindowSymbolic[wIdx].second = 0;

                    size_t flopsWindow = 0;

                    for(int k = 0; k < nmatrices; k++){
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                        auto first = rowIds->begin() + (*colPtr)[i];
                        auto last = rowIds->begin() + (*colPtr)[i+1];
                        size_t startIdx, endIdx, midIdx;

                        if(rowStart > 0){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowStart) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            first = rowIds->begin() + endIdx;
                            //first = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowStart );
                        }

                        if(rowEnd < nrows){
                            startIdx = (*colPtr)[i];
                            endIdx = (*colPtr)[i+1];
                            midIdx = (startIdx + endIdx) / 2;
                            while(startIdx < endIdx){
                                if((*rowIds)[midIdx] < rowEnd) startIdx = midIdx + 1;
                                else endIdx = midIdx;
                                midIdx = (startIdx + endIdx) / 2;
                            }
                            last = rowIds->begin() + endIdx;
                            //last = std::lower_bound( rowIds->begin() + (*colPtr)[i], rowIds->begin() + (*colPtr)[i+1], rowEnd );
                        }
                        rowIdsRange[tid * padding + k].first = first - rowIds->begin();
                        rowIdsRange[tid * padding + k].second = last - rowIds->begin();

                        flopsWindow += last-first;
                    }

                    size_t htSize = minHashTableSize;
                    while(htSize < flopsWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j){
                        globalHashVec[j] = -1;
                    }

                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();

                        auto first = rowIds->begin() + rowIdsRange[tid  * padding + k].first;
                        auto last = rowIds->begin() + rowIdsRange[tid * padding + k].second;
                        
                        for(; first<last; first++){
                            RIT key = *first;
                            RIT hash = (key*hashScale) & (htSize-1);
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash] == key) //key is found in hash table
                                {
                                    break;
                                }
                                else if (globalHashVec[hash] == -1) //key is not registered yet
                                {
                                    globalHashVec[hash] = key;
                                    nnzPerCol[i]++;
                                    nnzPerWindowSymbolic[wIdx].second++;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }
                        }
                    }
                    if (w == 0){
                        //nWindowPerCol[i] = 1;
                        runningSum = nnzPerWindowSymbolic[wIdx].second;
                    }
                    else{
                        if(runningSum + nnzPerWindowSymbolic[wIdx].second > maxHashTableSize){
                            nWindowPerCol[i]++;
                            runningSum = nnzPerWindowSymbolic[wIdx].second;
                        }
                        else{
                            runningSum = runningSum + nnzPerWindowSymbolic[wIdx].second;
                        }
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();
    //printf("%lf,", t1-t0);
#ifdef DEBUG
    printf("[Sliding Hash]\tTime for symbolic: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of time consumed by threads:\n");
    getStats<double>(ttimes, true);
#endif
    //printf("---\n");
    
    pvector<RIT> prefixSumWindow(ncols+1, 0);
    ParallelPrefixSum(nWindowPerCol, prefixSumWindow);

    //printf("[Sliding Hash]\tStats of number of windows:\n");
    //getStats<RIT>(nWindowPerCol, true);

    pvector< std::pair<RIT, RIT> > nnzPerWindow(prefixSumWindow[ncols]);
    
    t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for(CPT i = 0; i < ncols; i++){
        int64_t nwindows = nWindowPerColSymbolic[i];
        int64_t wsIdx = prefixSumWindowSymbolic[i];
        int64_t wcIdx = prefixSumWindow[i];
        nnzPerWindow[wcIdx].first = nnzPerWindowSymbolic[wsIdx].first;
        nnzPerWindow[wcIdx].second = nnzPerWindowSymbolic[wsIdx].second;
        for(size_t w=1; w < nwindows; w++){
            wsIdx = prefixSumWindowSymbolic[i] + w;
            if(nnzPerWindow[wcIdx].second + nnzPerWindowSymbolic[wsIdx].second > maxHashTableSize){
                wcIdx++;
                nnzPerWindow[wcIdx].first = nnzPerWindowSymbolic[wsIdx].first;
                nnzPerWindow[wcIdx].second = nnzPerWindowSymbolic[wsIdx].second;
            }
            else{
                nnzPerWindow[wcIdx].second = nnzPerWindow[wcIdx].second + nnzPerWindowSymbolic[wsIdx].second;
            }
        }
    }
    t1 = omp_get_wtime();

    pvector<CPT> prefixSum(ncols+1, 0);
    ParallelPrefixSum(nnzPerCol, prefixSum);

    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);
    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    CPT nnzCPerThreadExpected = nnzCTot / nthreads;
    CPT nnzCPerSplitExpected = nnzCTot / nsplits;
    
    pvector<double> colTimes(ncols);

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT> > globalHashVec(minHashTableSize);
#pragma omp for schedule(static)
        for(CPT i = 0; i < ncols; i++){
            //double tc = omp_get_wtime();
            RIT nwindows = nWindowPerCol[i];
            if(nwindows > 1){
                for(int k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    rowIdsRange[padding * tid + k].first = (*colPtr)[i];
                    rowIdsRange[padding * tid + k].second = (*colPtr)[i+1];
                }
                for (int w = 0; w < nwindows; w++){
                    RIT wIdx = prefixSumWindow[i] + w;
                    RIT rowStart = nnzPerWindow[wIdx].first;
                    RIT rowEnd = (w == nWindowPerCol[i]-1) ? nrows : nnzPerWindow[wIdx+1].first;
                    RIT nnzWindow = nnzPerWindow[wIdx].second;

                    size_t htSize = minHashTableSize;
                    while(htSize < nnzWindow) //htSize is set as 2^n
                    {
                        htSize <<= 1;
                    }
                    if(globalHashVec.size() < htSize)
                        globalHashVec.resize(htSize);
                    for(size_t j=0; j < htSize; ++j)
                    {
                        globalHashVec[j].first = -1;
                    }

                    for(int k = 0; k < nmatrices; k++) {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                        while( (rowIdsRange[padding * tid + k].first < rowIdsRange[padding * tid + k].second) && ((*rowIds)[rowIdsRange[padding * tid + k].first] < rowEnd) ){
                            RIT j = rowIdsRange[padding * tid + k].first;
                            RIT key = (*rowIds)[j];
                            RIT hash = (key*hashScale) & (htSize-1);
                            VT curval = (*nzVals)[j];
                            while (1) //hash probing
                            {
                                if (globalHashVec[hash].first == key) //key is found in hash table
                                {
                                    globalHashVec[hash].second += curval;
                                    break;
                                }
                                else if (globalHashVec[hash].first == -1) //key is not registered yet
                                {
                                    globalHashVec[hash].first = key;
                                    globalHashVec[hash].second = curval;
                                    break;
                                }
                                else //key is not found
                                {
                                    hash = (hash+1) & (htSize-1);
                                }
                            }

                            rowIdsRange[padding * tid + k].first++;
                        } 
                    }

                    if(sorted){
                        size_t index = 0;
                        for (size_t j=0; j < htSize; ++j){
                            if (globalHashVec[j].first != -1){
                                globalHashVec[index++] = globalHashVec[j];
                            }
                        }
                
                        integerSort<VT>(globalHashVec.data(), index);
                    
                        for (size_t j=0; j < index; ++j){
                            CrowIds[prefixSum[i]] = globalHashVec[j].first;
                            CnzVals[prefixSum[i]] = globalHashVec[j].second;
                            prefixSum[i] ++;
                        }
                    }
                    else{
                        for (size_t j=0; j < htSize; ++j)
                        {
                            if (globalHashVec[j].first != -1)
                            {
                                CrowIds[prefixSum[i]] = globalHashVec[j].first;
                                CnzVals[prefixSum[i]] = globalHashVec[j].second;
                                prefixSum[i] ++;
                            }
                        }
                    }
                }
                
            }
            else{
                RIT wIdx = prefixSumWindow[i];
                RIT nnzWindow = nnzPerWindow[wIdx].second;

                size_t htSize = minHashTableSize;
                while(htSize < nnzWindow) //htSize is set as 2^n
                {
                    htSize <<= 1;
                }
                if(globalHashVec.size() < htSize)
                    globalHashVec.resize(htSize);
                for(size_t j=0; j < htSize; ++j)
                {
                    globalHashVec[j].first = -1;
                }
                
                for(int k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                    for( RIT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++){
                        RIT key = (*rowIds)[j];
                        RIT hash = (key*hashScale) & (htSize-1);
                        VT curval = (*nzVals)[j];
                        while (1) //hash probing
                        {
                            if (globalHashVec[hash].first == key) //key is found in hash table
                            {
                                globalHashVec[hash].second += curval;
                                break;
                            }
                            else if (globalHashVec[hash].first == -1) //key is not registered yet
                            {
                                globalHashVec[hash].first = key;
                                globalHashVec[hash].second = curval;
                                break;
                            }
                            else //key is not found
                            {
                                hash = (hash+1) & (htSize-1);
                            }
                        }
                    } 
                }
                if(sorted){
                    size_t index = 0;
                    for (size_t j=0; j < htSize; ++j){
                        if (globalHashVec[j].first != -1){
                            globalHashVec[index++] = globalHashVec[j];
                        }
                    }
            
                    integerSort<VT>(globalHashVec.data(), index);
                
                    for (size_t j=0; j < index; ++j){
                        CrowIds[prefixSum[i]] = globalHashVec[j].first;
                        CnzVals[prefixSum[i]] = globalHashVec[j].second;
                        prefixSum[i] ++;
                    }
                }
                else{
                    for (size_t j=0; j < htSize; ++j)
                    {
                        if (globalHashVec[j].first != -1)
                        {
                            CrowIds[prefixSum[i]] = globalHashVec[j].first;
                            CnzVals[prefixSum[i]] = globalHashVec[j].second;
                            prefixSum[i] ++;
                        }
                    }
                }
            }
            //colTimes[i] = omp_get_wtime() - tc;
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();
    //printf("%lf,", t1-t0);

#ifdef DEBUG
    printf("[Sliding Hash]\tTime for computation: %lf\n", t1-t0);
    printf("[Sliding Hash]\tStats of parallel section timings:\n");
    getStats<double>(ttimes, true);
#endif
    //printf("***\n\n");

    //std::string ws = std::to_string(windowSize);
    //std::string prefix("window-");
    //std::string filename = prefix + ws;
    //std::fstream fs;
    //fs.open (filename, std::fstream::out);
    //fs << "colid,nnzc,nwindow,time" << std::endl;
    //for (CPT i = 0; i < ncols; i++){
        //fs << i << "," << nnzPerCol[i] << "," << nWindowPerCol[i] << "," << colTimes[i] << std::endl;
    //}

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);
    return std::move(sumMat);
}

/*
 * Sparse allocator based k-way sparse addition
 * using dynamic load balancing strategy
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddSpADynamic(std::vector<CSC<RIT, VT, CPT>* > & matrices)
{
    double t0, t1, t2, t3;

    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();

    pvector<CPT> nnzPerCol(ncols);
    pvector<CPT> flops_per_column(ncols, 0);

    t0 = omp_get_wtime();
#pragma omp parallel for
    for(CIT i = 0; i < ncols; i++){
        for(int k = 0; k < nmatrices; k++){
            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
            flops_per_column[i] += (*colPtr)[i+1] - (*colPtr)[i];
        }
    }
    t1 = omp_get_wtime();

    pvector<CPT> prefix_sum(ncols+1);
    ParallelPrefixSum(flops_per_column, prefix_sum);
    
    int64_t flops_tot = prefix_sum[ncols];
    int64_t flops_per_thread_expected;
    int64_t flops_per_split_expected;
    
    CIT nthreads;
    CIT nsplits;
    pvector<CIT> splitters;
    pvector<double> ttimes;
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector< RIT > globalHashVec(nrows, -1);
        std::vector< RIT > usedPos(nrows, -1);
        RIT usedPosIdx = 0;
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min( nthreads*4, ncols );
            splitters.resize(nsplits);
            ttimes.resize(nthreads);
            flops_per_thread_expected = flops_tot / nthreads;
            flops_per_split_expected = flops_tot / nsplits;
        }
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * flops_per_split_expected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++) {
            CIT colStart = splitters[s];
            CIT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CIT i = colStart; i < colEnd; i++){
                nnzPerCol[i] = 0;
            
                for(int k = 0; k < nmatrices; k++){
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                
                    for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                    {
                        RIT key = (*rowIds)[j];
                        if (globalHashVec[key] == -1) {
                            globalHashVec[key] = key;
                            nnzPerCol[i]++;
                            usedPos[usedPosIdx] = key;
                            usedPosIdx++;
                        }
                    }
                }

                // Re-initialize the hash table for next iteration
                for(size_t j=0; j < usedPosIdx; j++){
                    globalHashVec[usedPos[j]] = -1;
                }
                usedPosIdx=0;
            }
        }
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();

    ParallelPrefixSum(nnzPerCol, prefix_sum);
    
    pvector<CPT> CcolPtr(prefix_sum.begin(), prefix_sum.end());
    pvector<RIT> CrowIds(prefix_sum[ncols]);
    pvector<VT> CnzVals(prefix_sum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefix_sum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefix_sum[ncols];
    CPT nnzCPerThreadExpected = nnzCTot / nthreads;
    CPT nnzCPerSplitExpected = nnzCTot / nsplits;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT>> globalHashVec(nrows);
        for(size_t j=0; j < nrows; ++j) globalHashVec[j].first = -1;
        std::vector< RIT > usedPos(nrows, -1);
        RIT usedPosIdx = 0;
#pragma omp barrier
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefix_sum.begin(), prefix_sum.end(), s * nnzCPerSplitExpected) - prefix_sum.begin();
        }
#pragma omp for schedule(dynamic) 
        for(size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = s < nsplits-1 ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                if(nnzPerCol[i] != 0){
                    for(int k = 0; k < nmatrices; k++)
                    {
                        const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                        const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                        const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                    
                        for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                        {
                            RIT key = (*rowIds)[j];
                            VT curval = (*nzVals)[j];
                            if (globalHashVec[key].first == key) //key is found in hash table
                            {
                                globalHashVec[key].second += curval;
                            }
                            else if (globalHashVec[key].first == -1) //key is not registered yet
                            {
                                globalHashVec[key].first = key;
                                globalHashVec[key].second = curval;
                                usedPos[usedPosIdx] = key;
                                usedPosIdx++;
                            }
                        }
                    }
                    
                    integerSort(usedPos.data(), usedPosIdx);
               
                    for (size_t j=0; j < usedPosIdx; ++j){
                        CrowIds[prefix_sum[i]] = globalHashVec[usedPos[j]].first;
                        CnzVals[prefix_sum[i]] = globalHashVec[usedPos[j]].second;
                        prefix_sum[i]++;
                        globalHashVec[usedPos[j]].first = -1; // Re-initializing the hash table for next iteration
                    }
                    usedPosIdx=0; // Re-initializing the used array position for next iteration
                }
            }
        }  // parallel programming ended
        ttime = omp_get_wtime() - ttime;
        ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();

    Timer clock;
    clock.Start();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    clock.Stop();
    return std::move(sumMat);
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddSpAStatic(std::vector<CSC<RIT, VT, CPT>* > & matrices)
{
    double t0, t1, t2, t3;

    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();

    pvector<CPT> nnzPerCol(ncols);
    pvector<CPT> flops_per_column(ncols, 0);

    t0 = omp_get_wtime();
#pragma omp parallel for
    for(CIT i = 0; i < ncols; i++){
        for(int k = 0; k < nmatrices; k++){
            const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
            flops_per_column[i] += (*colPtr)[i+1] - (*colPtr)[i];
        }
    }
    t1 = omp_get_wtime();

    pvector<CPT> prefix_sum(ncols+1);
    ParallelPrefixSum(flops_per_column, prefix_sum);
    
    int64_t flops_tot = prefix_sum[ncols];
    int64_t flops_per_thread_expected;
    int64_t flops_per_split_expected;
    
    CIT nthreads;
    CIT nsplits;
    pvector<CIT> splitters;
    pvector<double> ttimes;
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        std::vector< RIT > globalHashVec(nrows);
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min( nthreads*4, ncols );
            splitters.resize(nsplits);
            ttimes.resize(nthreads);
            flops_per_thread_expected = flops_tot / nthreads;
            flops_per_split_expected = flops_tot / nsplits;
        }
#pragma omp for schedule(static)
        for(CIT i = 0; i < ncols; i++){
            nnzPerCol[i] = 0;
            for(size_t j=0; j < nrows; ++j) globalHashVec[j] = -1;
        
            for(int k = 0; k < nmatrices; k++){
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                const pvector<VT> *nzVals = matrices[k]->get_nzVals();
            
                for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                {
                    RIT key = (*rowIds)[j];
                    if (globalHashVec[key] == -1) {
                        globalHashVec[key] = key;
                        nnzPerCol[i]++;
                    }
                }
            }
        }
        ttime = omp_get_wtime() - ttime;
        //ttimes[tid] = ttime;
    }
    t1 = omp_get_wtime();

    ParallelPrefixSum(nnzPerCol, prefix_sum);
    
    pvector<CPT> CcolPtr(prefix_sum.begin(), prefix_sum.end());
    pvector<RIT> CrowIds(prefix_sum[ncols]);
    pvector<VT> CnzVals(prefix_sum[ncols]);

    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefix_sum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefix_sum[ncols];
    CPT nnzCPerThreadExpected = nnzCTot / nthreads;
    CPT nnzCPerSplitExpected = nnzCTot / nsplits;

    t0 = omp_get_wtime();
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        int tid = omp_get_thread_num();
        std::vector< std::pair<RIT,VT>> globalHashVec(nrows);
#pragma omp for schedule(static) 
        for(CPT i = 0; i < ncols; i++){
            if(nnzPerCol[i] != 0){
                for(size_t j=0; j < nrows; ++j) globalHashVec[j].first = -1;
            
                for(int k = 0; k < nmatrices; k++)
                {
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();
                
                    for(CPT j = (*colPtr)[i]; j < (*colPtr)[i+1]; j++)
                    {
                        RIT key = (*rowIds)[j];
                        VT curval = (*nzVals)[j];
                        if (globalHashVec[key].first == key) //key is found in hash table
                        {
                            globalHashVec[key].second += curval;
                        }
                        else if (globalHashVec[key].first == -1) //key is not registered yet
                        {
                            globalHashVec[key].first = key;
                            globalHashVec[key].second = curval;
                        }
                    }
                }
           
                for (size_t j=0; j < nrows; ++j){
                    if (globalHashVec[j].first != -1){
                        CrowIds[prefix_sum[i]] = globalHashVec[j].first;
                        CnzVals[prefix_sum[i]] = globalHashVec[j].second;
                        prefix_sum[i]++;
                    }
                }
            }
        } // parallel programming ended
        ttime = omp_get_wtime() - ttime;
        //ttimes[tid] = ttime;
#pragma omp barrier
    }
    t1 = omp_get_wtime();

    Timer clock;
    clock.Start();

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);

    clock.Stop();
    return std::move(sumMat);
}

/*
 *  Estimates nnumber of non-zeroes when adding two CSC matrices in regular way
 *  Assumes that entries of each column are sorted according to the order of row id
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
pvector<RIT> symbolicSpAddRegularDynamic(CSC<RIT, VT, CPT>* A, CSC<RIT, VT, CPT>* B)
{
    double t0, t1, t3, t4;

    CIT nthreads;
    CIT nsplits;
    CIT ncols = A->get_ncols();
    RIT nrows = A->get_nrows();

    const pvector<CPT> *AcolPtr = A->get_colPtr();
    const pvector<RIT> *ArowIds = A->get_rowIds();
    const pvector<VT> *AnzVals = A->get_nzVals();
    const pvector<CPT> *BcolPtr = B->get_colPtr();
    const pvector<RIT> *BrowIds = B->get_rowIds();
    const pvector<VT> *BnzVals = B->get_nzVals();
    
    pvector<RIT> nnzCPerCol(ncols);

    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<size_t> flopsPerCol(ncols);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
        }
#pragma omp for
        for(CPT i = 0; i < ncols; i++){
            flopsPerCol[i] = (*AcolPtr)[i+1] - (*AcolPtr)[i];
            flopsPerCol[i] += (*BcolPtr)[i+1] - (*BcolPtr)[i];
        }
    }

    pvector<size_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);
    
    size_t flopsTot = prefixSumSymbolic[ncols];
    size_t flopsPerThreadExpected;
    size_t flopsPerSplitExpected;

    ttimes.resize(nthreads);
    splitters.resize(nsplits);
    flopsPerSplitExpected = flopsTot / nsplits;
    flopsPerThreadExpected = flopsTot / nthreads;
    
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        size_t tid = omp_get_thread_num();
#pragma omp for
        for(size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSumSymbolic.begin(), prefixSumSymbolic.end(), s * flopsPerSplitExpected) - prefixSumSymbolic.begin();
        }
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                RIT ArowsStart = (*AcolPtr)[i];
                RIT ArowsEnd = (*AcolPtr)[i+1];
                RIT BrowsStart = (*BcolPtr)[i];
                RIT BrowsEnd = (*BcolPtr)[i+1];
                RIT Aptr = ArowsStart;
                RIT Bptr = BrowsStart;
                nnzCPerCol[i] = 0;
                
                while (Aptr < ArowsEnd || Bptr < BrowsEnd){
                    if (Aptr >= ArowsEnd){
                        // Entries of A has finished
                        // Increment nnzCPerCol[i]
                        // Increment BPtr
                        nnzCPerCol[i]++;
                        Bptr++;
                    }
                    else if (Bptr >= BrowsEnd){
                        // Entries of B has finished
                        // Increment nnzCPerCol[i]
                        // Increment APtr
                        nnzCPerCol[i]++;
                        Aptr++;
                    }
                    else {
                        if ( (*ArowIds)[Aptr] < (*BrowIds)[Bptr]){
                            // Increment nnzCPerCol[i]
                            // Increment APtr 
                            nnzCPerCol[i]++;
                            Aptr++;
                        }
                        else if ((*ArowIds)[Aptr] > (*BrowIds)[Bptr]){
                            // Increment nnzCPerCol[i]     
                            // Increment BPtr
                            nnzCPerCol[i]++;
                            Bptr++;
                        }
                        else{
                            // Increment nnzCPerCol[i]
                            // Increment APtr, BPtr 
                            nnzCPerCol[i]++;
                            Aptr++;
                            Bptr++;
                        }
                    }
                }
            }
        }
    }

    return std::move(nnzCPerCol);
}


/*
 *  Estimates nnumber of non-zeroes when adding two CSC matrices in regular way
 *  Assumes that entries of each column are sorted according to the order of row id
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
pvector<RIT> symbolicSpAddRegularStatic(CSC<RIT, VT, CPT>* A, CSC<RIT, VT, CPT>* B)
{
    double t0, t1, t3, t4;

    CIT nthreads;
    CIT nsplits;
    CIT ncols = A->get_ncols();
    RIT nrows = A->get_nrows();

    const pvector<CPT> *AcolPtr = A->get_colPtr();
    const pvector<RIT> *ArowIds = A->get_rowIds();
    const pvector<VT> *AnzVals = A->get_nzVals();
    const pvector<CPT> *BcolPtr = B->get_colPtr();
    const pvector<RIT> *BrowIds = B->get_rowIds();
    const pvector<VT> *BnzVals = B->get_nzVals();
    
    pvector<RIT> nnzCPerCol(ncols);
    pvector<size_t> flopsPerCol(ncols);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
        }
#pragma omp for
        for(CPT i = 0; i < ncols; i++){
            flopsPerCol[i] = (*AcolPtr)[i+1] - (*AcolPtr)[i];
            flopsPerCol[i] += (*BcolPtr)[i+1] - (*BcolPtr)[i];
        }
    }

    pvector<size_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);
    
    size_t flopsTot = prefixSumSymbolic[ncols];
    size_t flopsPerThreadExpected;
    size_t flopsPerSplitExpected;
    
#pragma omp parallel
    {
        double ttime = omp_get_wtime();
        size_t tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for(CPT i = 0; i < ncols; i++){
            RIT ArowsStart = (*AcolPtr)[i];
            RIT ArowsEnd = (*AcolPtr)[i+1];
            RIT BrowsStart = (*BcolPtr)[i];
            RIT BrowsEnd = (*BcolPtr)[i+1];
            RIT Aptr = ArowsStart;
            RIT Bptr = BrowsStart;
            nnzCPerCol[i] = 0;
            
            while (Aptr < ArowsEnd || Bptr < BrowsEnd){
                if (Aptr >= ArowsEnd){
                    // Entries of A has finished
                    // Increment nnzCPerCol[i]
                    // Increment BPtr
                    nnzCPerCol[i]++;
                    Bptr++;
                }
                else if (Bptr >= BrowsEnd){
                    // Entries of B has finished
                    // Increment nnzCPerCol[i]
                    // Increment APtr
                    nnzCPerCol[i]++;
                    Aptr++;
                }
                else {
                    if ( (*ArowIds)[Aptr] < (*BrowIds)[Bptr]){
                        // Increment nnzCPerCol[i]
                        // Increment APtr 
                        nnzCPerCol[i]++;
                        Aptr++;
                    }
                    else if ((*ArowIds)[Aptr] > (*BrowIds)[Bptr]){
                        // Increment nnzCPerCol[i]     
                        // Increment BPtr
                        nnzCPerCol[i]++;
                        Bptr++;
                    }
                    else{
                        // Increment nnzCPerCol[i]
                        // Increment APtr, BPtr 
                        nnzCPerCol[i]++;
                        Aptr++;
                        Bptr++;
                    }
                }
            }
        }
    }

    return std::move(nnzCPerCol);
}

/*
 *  Adds two CSC matrices in regular way (the way merge operation of MergeSort works)
 *  Assumes that entries of each column are sorted according to the order of row id
 *  Assumes that all sanity checks are done before, so do not perform any
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpAddRegularDynamic(CSC<RIT, VT, CPT>* A, CSC<RIT, VT, CPT>* B, pvector<RIT> & nnzCPerCol)
{
    double t0, t1, t3, t4;

    CIT nthreads;
    CIT nsplits;
    CIT ncols = A->get_ncols();
    RIT nrows = A->get_nrows();

    const pvector<CPT> *AcolPtr = A->get_colPtr();
    const pvector<RIT> *ArowIds = A->get_rowIds();
    const pvector<VT> *AnzVals = A->get_nzVals();
    const pvector<CPT> *BcolPtr = B->get_colPtr();
    const pvector<RIT> *BrowIds = B->get_rowIds();
    const pvector<VT> *BnzVals = B->get_nzVals();

    pvector<CPT> prefixSum(ncols+1);
    ParallelPrefixSum(nnzCPerCol, prefixSum);
    CSC<RIT, VT, CPT> C(nrows, ncols, prefixSum[ncols], false, true);
    
    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);
    
    CPT nnzCTot = prefixSum[ncols];
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;
    CPT nnzCPerThreadExpected;
    CPT nnzCPerSplitExpected;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
            nnzCPerSplitExpected = nnzCTot / nsplits;
            splitters.resize(nsplits);
        }
#pragma omp barrier
#pragma omp for
        for(size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSum.begin(), prefixSum.end(), s * nnzCPerSplitExpected) - prefixSum.begin();
        }
#pragma omp barrier
#pragma omp for schedule(dynamic)
        for(size_t s = 0; s < nsplits; s++){
            CPT colStart = splitters[s];
            CPT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CPT i = colStart; i < colEnd; i++){
                RIT ArowsStart = (*AcolPtr)[i];
                RIT ArowsEnd = (*AcolPtr)[i+1];
                RIT BrowsStart = (*BcolPtr)[i];
                RIT BrowsEnd = (*BcolPtr)[i+1];
                RIT Aptr = ArowsStart;
                RIT Bptr = BrowsStart;
                RIT Cptr = prefixSum[i];

                while (Aptr < ArowsEnd || Bptr < BrowsEnd){
                    if (Aptr >= ArowsEnd){
                        // Entries of A has finished
                        // Copy the entry of BPtr to the CPtr
                        // Increment BPtr and CPtr
                        CrowIds[Cptr] = (*BrowIds)[Bptr];
                        CnzVals[Cptr] = (*BnzVals)[Bptr];
                        Bptr++;
                        Cptr++;
                    }
                    else if (Bptr >= BrowsEnd){
                        // Entries of B has finished
                        // Copy the entry of APtr to the CPtr
                        // Increment APtr and CPtr
                        CrowIds[Cptr] = (*ArowIds)[Aptr];
                        CnzVals[Cptr] = (*AnzVals)[Aptr];
                        Aptr++;
                        Cptr++;
                    }
                    else {
                        if ( (*ArowIds)[Aptr] < (*BrowIds)[Bptr]){
                            // Copy the entry of APtr to the CPtr
                            // Increment APtr and CPtr
                            CrowIds[Cptr] = (*ArowIds)[Aptr];
                            CnzVals[Cptr] = (*AnzVals)[Aptr];
                            Aptr++;
                            Cptr++;
                        }
                        else if ((*ArowIds)[Aptr] > (*BrowIds)[Bptr]){
                            // Copy the entry of BPtr to the CPtr
                            // Increment BPtr and CPtr
                            CrowIds[Cptr] = (*BrowIds)[Bptr];
                            CnzVals[Cptr] = (*BnzVals)[Bptr];
                            Bptr++;
                            Cptr++;
                        }
                        else{
                            // Sum the entries of APtr and BPtr then store at CPtr
                            // Increment APtr, BPtr and CPtr
                            CrowIds[Cptr] = (*ArowIds)[Aptr];
                            CnzVals[Cptr] = (*AnzVals)[Aptr] + (*BnzVals)[Bptr];
                            Aptr++;
                            Bptr++;
                            Cptr++;
                        }
                    }
                }
                //nnzPerThread[tid] += (ArowsEnd - ArowsStart) + (BrowsEnd - BrowsStart);
            }
        }
    }

    C.cols_pvector(&CcolPtr);
    C.nz_rows_pvector(&CrowIds);
    C.nz_vals_pvector(&CnzVals);

    return std::move(C);
}

/*
 *  Adds two CSC matrices in regular way (the way merge operation of MergeSort works)
 *  Assumes that entries of each column are sorted according to the order of row id
 *  Assumes that all sanity checks are done before, so do not perform any
 * */
template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpAddRegularStatic(CSC<RIT, VT, CPT>* A, CSC<RIT, VT, CPT>* B, pvector<RIT> & nnzCPerCol)
{
    double t0, t1, t3, t4;

    CIT nthreads;
    CIT nsplits;
    CIT ncols = A->get_ncols();
    RIT nrows = A->get_nrows();

    const pvector<CPT> *AcolPtr = A->get_colPtr();
    const pvector<RIT> *ArowIds = A->get_rowIds();
    const pvector<VT> *AnzVals = A->get_nzVals();
    const pvector<CPT> *BcolPtr = B->get_colPtr();
    const pvector<RIT> *BrowIds = B->get_rowIds();
    const pvector<VT> *BnzVals = B->get_nzVals();

    pvector<CPT> prefixSum(ncols+1);
    ParallelPrefixSum(nnzCPerCol, prefixSum);
    CSC<RIT, VT, CPT> C(nrows, ncols, prefixSum[ncols], false, true);
    
    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);
    
    CPT nnzCTot = prefixSum[ncols];
    pvector<double> ttimes; // To record time taken by each thread

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for(CPT i = 0; i < ncols; i++){
            RIT ArowsStart = (*AcolPtr)[i];
            RIT ArowsEnd = (*AcolPtr)[i+1];
            RIT BrowsStart = (*BcolPtr)[i];
            RIT BrowsEnd = (*BcolPtr)[i+1];
            RIT Aptr = ArowsStart;
            RIT Bptr = BrowsStart;
            RIT Cptr = prefixSum[i];

            while (Aptr < ArowsEnd || Bptr < BrowsEnd){
                if (Aptr >= ArowsEnd){
                    // Entries of A has finished
                    // Copy the entry of BPtr to the CPtr
                    // Increment BPtr and CPtr
                    CrowIds[Cptr] = (*BrowIds)[Bptr];
                    CnzVals[Cptr] = (*BnzVals)[Bptr];
                    Bptr++;
                    Cptr++;
                }
                else if (Bptr >= BrowsEnd){
                    // Entries of B has finished
                    // Copy the entry of APtr to the CPtr
                    // Increment APtr and CPtr
                    CrowIds[Cptr] = (*ArowIds)[Aptr];
                    CnzVals[Cptr] = (*AnzVals)[Aptr];
                    Aptr++;
                    Cptr++;
                }
                else {
                    if ( (*ArowIds)[Aptr] < (*BrowIds)[Bptr]){
                        // Copy the entry of APtr to the CPtr
                        // Increment APtr and CPtr
                        CrowIds[Cptr] = (*ArowIds)[Aptr];
                        CnzVals[Cptr] = (*AnzVals)[Aptr];
                        Aptr++;
                        Cptr++;
                    }
                    else if ((*ArowIds)[Aptr] > (*BrowIds)[Bptr]){
                        // Copy the entry of BPtr to the CPtr
                        // Increment BPtr and CPtr
                        CrowIds[Cptr] = (*BrowIds)[Bptr];
                        CnzVals[Cptr] = (*BnzVals)[Bptr];
                        Bptr++;
                        Cptr++;
                    }
                    else{
                        // Sum the entries of APtr and BPtr then store at CPtr
                        // Increment APtr, BPtr and CPtr
                        CrowIds[Cptr] = (*ArowIds)[Aptr];
                        CnzVals[Cptr] = (*AnzVals)[Aptr] + (*BnzVals)[Bptr];
                        Aptr++;
                        Bptr++;
                        Cptr++;
                    }
                }
            }
            //nnzPerThread[tid] += (ArowsEnd - ArowsStart) + (BrowsEnd - BrowsStart);
        }
    }

    C.cols_pvector(&CcolPtr);
    C.nz_rows_pvector(&CrowIds);
    C.nz_vals_pvector(&CnzVals);

    return std::move(C);
}

// MUST DO: Need to use template.
bool compare(std::tuple<uint32_t, float, int> x, std::tuple<uint32_t, float, int> y){
    if(std::get<0>(x) < std::get<0>(y)) return false;
    else return true;
}

//template <typename RIT, typename VT>
//bool compare(std::tuple<size_t, RIT, VT> x, std::tuple<size_t, RIT, VT> y){
    //if(std::get<0>(x) > std::get<0>(y)) return false;
    //else return true;
//}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHeapDynamic(std::vector<CSC<RIT, VT, CPT>* > &matrices){

    double t0, t1, t2, t3, t4, t5;

    t0 = omp_get_wtime();
    CIT nthreads;
    CIT nsplits;
    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<size_t> flopsPerCol(ncols);
    pvector<RIT> nnzPerCol(ncols, 0);

    t2 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
        }
#pragma omp for
        for(CPT i = 0; i < ncols; i++){
            flopsPerCol[i] = 0;
            for(int k = 0; k < nmatrices; k++){
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                flopsPerCol[i] += (*colPtr)[i+1] - (*colPtr)[i];
            }
        }
    }
    t3 = omp_get_wtime();

    pvector<size_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);
    
    size_t flopsTot = prefixSumSymbolic[ncols];
    size_t flopsPerThreadExpected;
    size_t flopsPerSplitExpected;

    ttimes.resize(nthreads);
    splitters.resize(nsplits);
    flopsPerSplitExpected = flopsTot / nsplits;
    flopsPerThreadExpected = flopsTot / nthreads;
    
    t1 = omp_get_wtime();
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector< std::tuple<RIT, VT, int> > heap(nmatrices);
        std::vector< CPT > curptr(nmatrices, static_cast<CPT>(0));
        CPT hsize = 0;
#pragma omp for
        for (size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSumSymbolic.begin(), prefixSumSymbolic.end(), s * flopsPerSplitExpected) - prefixSumSymbolic.begin();
        }
#pragma omp barrier
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++) {
            CIT colStart = splitters[s];
            CIT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CIT i = colStart; i < colEnd; i++){
                hsize = 0;
                for(int k = 0; k < nmatrices; k++){
                    curptr[k] = 0;
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                    RIT j = (*colPtr)[i] + curptr[k];
                    if(j < (*colPtr)[i+1]){
                        heap[hsize] = std::make_tuple((*rowIds)[j], (*nzVals)[j], k);
                        hsize++;
                        curptr[k]++;
                    }
                }

                std::make_heap(heap.data(), heap.data()+hsize, compare);

                nnzPerCol[i] = 0;
                RIT lastPoppedRow = -1;
                while(hsize > 0){
                    std::pop_heap(heap.data(), heap.data()+hsize, compare);

                    int k = std::get<2>(heap[hsize-1]);
                    RIT r = std::get<0>(heap[hsize-1]);

                    if(r != lastPoppedRow){
                        nnzPerCol[i]++;
                        lastPoppedRow = r;
                    }

                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                    RIT j = (*colPtr)[i] + curptr[k];
                    if(j < (*colPtr)[i+1]){
                        heap[hsize-1] = std::make_tuple((*rowIds)[j], (*nzVals)[j], k);
                        std::push_heap(heap.data(), heap.data()+hsize, compare);
                        curptr[k]++;
                    }
                    else{
                        hsize--;
                    }
                }
            }
        }
    }

    pvector<size_t> prefixSum(ncols+1, 0);
    ParallelPrefixSum(nnzPerCol, prefixSum);

    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);
    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    CPT nnzCPerThreadExpected = nnzCTot / nthreads;
    CPT nnzCPerSplitExpected = nnzCTot / nsplits;
    
    pvector<double> colTimes(ncols);
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector< std::tuple<RIT, VT, int> > heap(nmatrices);
        std::vector< CPT > curptr(nmatrices, static_cast<CPT>(0));
        CPT hsize = 0;
#pragma omp for
        for(size_t s = 0; s < nsplits; s++){
            splitters[s] = std::lower_bound(prefixSum.begin(), prefixSum.end(), s * nnzCPerSplitExpected) - prefixSum.begin();
        }
#pragma omp barrier
#pragma omp for schedule(dynamic)
        for (size_t s = 0; s < nsplits; s++) {
            CIT colStart = splitters[s];
            CIT colEnd = (s < nsplits-1) ? splitters[s+1] : ncols;
            for(CIT i = colStart; i < colEnd; i++){
                hsize = 0;
                for(int k = 0; k < nmatrices; k++){
                    curptr[k] = 0;
                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                    RIT j = (*colPtr)[i] + curptr[k];
                    if(j < (*colPtr)[i+1]){
                        heap[hsize] = std::make_tuple((*rowIds)[j], (*nzVals)[j], k);
                        hsize++;
                        curptr[k]++;
                    }
                }

                std::make_heap(heap.data(), heap.data()+hsize, compare);

                nnzPerCol[i] = 0;
                RIT lastPoppedRow = -1;
                CPT idx = -1;
                while(hsize > 0){
                    std::pop_heap(heap.data(), heap.data()+hsize, compare);

                    int k = std::get<2>(heap[hsize-1]);
                    RIT r = std::get<0>(heap[hsize-1]);
                    VT v = std::get<1>(heap[hsize-1]);

                    if(r != lastPoppedRow){
                        lastPoppedRow = r;
                        CrowIds[prefixSum[i]] = r;
                        CnzVals[prefixSum[i]] = v;
                        prefixSum[i]++;
                    }
                    else{
                        CnzVals[prefixSum[i]-1] += v;
                    }

                    const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                    const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                    const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                    RIT j = (*colPtr)[i] + curptr[k];
                    if(j < (*colPtr)[i+1]){
                        heap[hsize-1] = std::make_tuple((*rowIds)[j], (*nzVals)[j], k);
                        std::push_heap(heap.data(), heap.data()+hsize, compare);
                        curptr[k]++;
                    }
                    else{
                        hsize--;
                    }
                }
            }
        }
    }

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);
    return std::move(sumMat);
}

template <typename RIT, typename CIT, typename VT= long double, typename CPT=size_t>
CSC<RIT, VT, CPT> SpMultiAddHeapStatic(std::vector<CSC<RIT, VT, CPT>* > &matrices){

    double t0, t1, t2, t3, t4, t5;

    t0 = omp_get_wtime();
    CIT nthreads;
    CIT nsplits;
    size_t nmatrices = matrices.size();
    CIT ncols = matrices[0]->get_ncols();
    RIT nrows = matrices[0]->get_nrows();
    
    pvector<double> ttimes; // To record time taken by each thread
    pvector<CPT> splitters; // To store load balance friendly split of columns accross threads;

    pvector<size_t> flopsPerCol(ncols);
    pvector<RIT> nnzPerCol(ncols, 0);

    t2 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if(tid == 0){
            nthreads = omp_get_num_threads();
            nsplits = std::min(nthreads * 4, ncols); // More split for better load balance
        }
#pragma omp for
        for(CPT i = 0; i < ncols; i++){
            flopsPerCol[i] = 0;
            for(int k = 0; k < nmatrices; k++){
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                flopsPerCol[i] += (*colPtr)[i+1] - (*colPtr)[i];
            }
        }
    }
    t3 = omp_get_wtime();

    pvector<size_t> prefixSumSymbolic(ncols+1, 0);
    ParallelPrefixSum(flopsPerCol, prefixSumSymbolic);
    
    size_t flopsTot = prefixSumSymbolic[ncols];
    size_t flopsPerThreadExpected;
    size_t flopsPerSplitExpected;

    ttimes.resize(nthreads);
    splitters.resize(nsplits);
    flopsPerSplitExpected = flopsTot / nsplits;
    flopsPerThreadExpected = flopsTot / nthreads;
    
    t1 = omp_get_wtime();
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector< std::tuple<RIT, VT, int> > heap(nmatrices);
        std::vector< CPT > curptr(nmatrices, static_cast<CPT>(0));
        CPT hsize = 0;
#pragma omp for schedule(static)
        for(CIT i = 0; i < ncols; i++){
            hsize = 0;
            for(int k = 0; k < nmatrices; k++){
                curptr[k] = 0;
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                RIT j = (*colPtr)[i] + curptr[k];
                if(j < (*colPtr)[i+1]){
                    heap[hsize] = std::make_tuple((*rowIds)[j], (*nzVals)[j], k);
                    hsize++;
                    curptr[k]++;
                }
            }

            std::make_heap(heap.data(), heap.data()+hsize, compare);

            nnzPerCol[i] = 0;
            RIT lastPoppedRow = -1;
            while(hsize > 0){
                std::pop_heap(heap.data(), heap.data()+hsize, compare);

                int k = std::get<2>(heap[hsize-1]);
                RIT r = std::get<0>(heap[hsize-1]);

                if(r != lastPoppedRow){
                    nnzPerCol[i]++;
                    lastPoppedRow = r;
                }

                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                RIT j = (*colPtr)[i] + curptr[k];
                if(j < (*colPtr)[i+1]){
                    heap[hsize-1] = std::make_tuple((*rowIds)[j], (*nzVals)[j], k);
                    std::push_heap(heap.data(), heap.data()+hsize, compare);
                    curptr[k]++;
                }
                else{
                    hsize--;
                }
            }
        }
    }

    pvector<size_t> prefixSum(ncols+1, 0);
    ParallelPrefixSum(nnzPerCol, prefixSum);

    pvector<CPT> CcolPtr(prefixSum.begin(), prefixSum.end());
    pvector<RIT> CrowIds(prefixSum[ncols]);
    pvector<VT> CnzVals(prefixSum[ncols]);
    CSC<RIT, VT, CPT> sumMat(nrows, ncols, prefixSum[ncols], false, true);
    sumMat.cols_pvector(&CcolPtr);
    
    CPT nnzCTot = prefixSum[ncols];
    CPT nnzCPerThreadExpected = nnzCTot / nthreads;
    CPT nnzCPerSplitExpected = nnzCTot / nsplits;
    
    pvector<double> colTimes(ncols);
    
    t0 = omp_get_wtime();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector< std::tuple<RIT, VT, int> > heap(nmatrices);
        std::vector< CPT > curptr(nmatrices, static_cast<CPT>(0));
        CPT hsize = 0;
#pragma omp for schedule(static)
        for(CIT i = 0; i < ncols; i++){
            hsize = 0;
            for(int k = 0; k < nmatrices; k++){
                curptr[k] = 0;
                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                RIT j = (*colPtr)[i] + curptr[k];
                if(j < (*colPtr)[i+1]){
                    heap[hsize] = std::make_tuple((*rowIds)[j], (*nzVals)[j], k);
                    hsize++;
                    curptr[k]++;
                }
            }

            std::make_heap(heap.data(), heap.data()+hsize, compare);

            nnzPerCol[i] = 0;
            RIT lastPoppedRow = -1;
            CPT idx = -1;
            while(hsize > 0){
                std::pop_heap(heap.data(), heap.data()+hsize, compare);

                int k = std::get<2>(heap[hsize-1]);
                RIT r = std::get<0>(heap[hsize-1]);
                VT v = std::get<1>(heap[hsize-1]);

                if(r != lastPoppedRow){
                    lastPoppedRow = r;
                    CrowIds[prefixSum[i]] = r;
                    CnzVals[prefixSum[i]] = v;
                    prefixSum[i]++;
                }
                else{
                    CnzVals[prefixSum[i]-1] += v;
                }

                const pvector<CPT> *colPtr = matrices[k]->get_colPtr();
                const pvector<RIT> *rowIds = matrices[k]->get_rowIds();
                const pvector<VT> *nzVals = matrices[k]->get_nzVals();

                RIT j = (*colPtr)[i] + curptr[k];
                if(j < (*colPtr)[i+1]){
                    heap[hsize-1] = std::make_tuple((*rowIds)[j], (*nzVals)[j], k);
                    std::push_heap(heap.data(), heap.data()+hsize, compare);
                    curptr[k]++;
                }
                else{
                    hsize--;
                }
            }
        }
    }

    sumMat.nz_rows_pvector(&CrowIds);
    sumMat.nz_vals_pvector(&CnzVals);
    return std::move(sumMat);
}

#endif
