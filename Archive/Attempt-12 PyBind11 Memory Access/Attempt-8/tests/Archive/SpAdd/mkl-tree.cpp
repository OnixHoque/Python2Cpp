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

#include "mkl.h"
#include "mkl_spblas.h"

#if defined ( MKL_ILP64 )
#define MKL_INT long long int /* 64 bit integer for large arrays. */
#else
#define MKL_INT int /* 32 bit integer for arrays < 2^31-1 elements. */
#endif

int main(int argc, char* argv[]){
    int x = atoi(argv[1]); // scale of random matrix, indicates number of rows
    int y = atoi(argv[2]); // scale of random matrix, indicates number of columns
    int d = atoi(argv[3]); // average degree of random matrix
	bool weighted = true;

	int k = atoi(argv[4]); // number of matrices
    int type = atoi(argv[5]); // Type of matrix
    int t = atoi(argv[6]); // number of threads

	std::vector< CSC<uint32_t, float, uint32_t>* > vec;
    std::vector< CSC<uint32_t, float, uint32_t>* > vec_temp;

    uint64_t total_nnz_in = 0;
    uint64_t total_nnz_out = 0;

    for(int i = 0; i < k; i++){
        COO<uint32_t, uint32_t, float> coo;
        double t0, t1;
        if(type == 0){
            coo.GenER(x, y, d, weighted, i);   // Generate a weighted ER matrix with 2^x rows, 2^y columns and d nonzeros per column using random seed i
            vec.push_back(new CSC<uint32_t, float, uint32_t>(coo));
        }
        else if(type == 1){
            // For RMAT matrix need to be square. So x need to be equal to y.
            if (x != y){
                int z = std::min(x,y);
                x = std::max(x,y);
                if(i == 0){
                    coo.GenRMAT(x, d, weighted, i);   // Generate a weighted RMAT matrix with 2^x rows, 2^x columns and d nonzeros per column using random seed i
                    CSC<uint32_t, float, uint32_t> m = CSC<uint32_t, float, uint32_t>(coo);
                    m.column_split(vec, k);
                }
            }
            else{
                coo.GenRMAT(x, d, weighted, i);   // Generate a weighted RMAT matrix with 2^x rows, 2^x columns and d nonzeros per column using random seed i
                vec.push_back(new CSC<uint32_t, float, uint32_t>(coo));
            }
        }
        else if(type == 2){
            std::string filename(argv[7]);
            filename = filename + std::to_string(i);
            t0 = omp_get_wtime();
            coo.ReadMM(filename);
            t1 = omp_get_wtime();
            vec.push_back(new CSC<uint32_t, float, uint32_t>(coo));
            //printf("Time to read %s: %lf seconds\n", filename.c_str(), t1-t0);
        }

        total_nnz_in += vec[i]->get_nnz();
    }
    
    Timer clock;

    // MKL specific codes
    
    double** mkl_values = (double**) malloc( k * sizeof(double*) );
    MKL_INT** mkl_rows = (MKL_INT**) malloc( k * sizeof(MKL_INT*) );
    MKL_INT** mkl_pointerB = (MKL_INT**) malloc( k * sizeof(MKL_INT*) );
    MKL_INT** mkl_pointerE = (MKL_INT**) malloc( k * sizeof(MKL_INT*) );

    sparse_matrix_t* mkl_csc_matrices = (sparse_matrix_t*) malloc( k * sizeof(sparse_matrix_t) );
    sparse_matrix_t* mkl_csr_matrices = (sparse_matrix_t*) malloc( k * sizeof(sparse_matrix_t) );
    sparse_matrix_t* mkl_sums = (sparse_matrix_t*) malloc( k * sizeof(sparse_matrix_t) );
    
    for(int i = 0; i < k; i++){
        mkl_values[i] = NULL;
        mkl_rows[i] = NULL;
        mkl_pointerB[i] = NULL;
        mkl_pointerE[i] = NULL;
        mkl_csc_matrices[i] = NULL;
        mkl_csr_matrices[i] = NULL;
        mkl_sums[i] = NULL;
    }

    for(int i = 0; i < k; i++){
        auto csc_nzVals = vec[i]->get_nzVals(); 
        mkl_values[i] = (double*) malloc( ( csc_nzVals->size() ) * sizeof(double) );
        for(int j = 0; j < csc_nzVals->size(); j++){
            mkl_values[i][j] = (double) (*csc_nzVals)[j];
        }

        auto csc_rowIds = vec[i]->get_rowIds(); 
        mkl_rows[i] = (MKL_INT*) malloc( ( csc_rowIds->size() ) * sizeof(MKL_INT) );
        for(int j = 0; j < csc_rowIds->size(); j++){
            mkl_rows[i][j] = (MKL_INT) (*csc_rowIds)[j];
        }

        auto csc_colPtr = vec[i]->get_colPtr();
        mkl_pointerB[i] = (MKL_INT*) malloc( ( csc_colPtr->size() ) * sizeof(MKL_INT) );
        mkl_pointerE[i] = (MKL_INT*) malloc( ( csc_colPtr->size() ) * sizeof(MKL_INT) );

        for(int j = 0; j < csc_colPtr->size(); j++){
            if(j == 0){
                mkl_pointerB[i][j] = (MKL_INT) (*csc_colPtr)[j];
            }
            else if(j == csc_colPtr->size()-1){
                mkl_pointerE[i][j-1] = (MKL_INT) (*csc_colPtr)[j];
            }
            else{
                mkl_pointerB[i][j] = (MKL_INT) (*csc_colPtr)[j];
                mkl_pointerE[i][j-1] = (MKL_INT) (*csc_colPtr)[j];
            }
        }

        //printf("Creating MKL CSR matrix %d: ", i);
        sparse_status_t create_status = mkl_sparse_d_create_csc (
            &(mkl_csc_matrices[i]), 
            SPARSE_INDEX_BASE_ZERO, 
            vec[i]->get_nrows(), 
            vec[i]->get_ncols(), 
            mkl_pointerB[i], 
            mkl_pointerE[i], 
            mkl_rows[i], 
            mkl_values[i]
        );
        //switch(create_status){
            //case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            //case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            //case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            //case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            //case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            //case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            //case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        //}
        //printf("\n");
        

        /**/
        //printf("Converting MKL CSC matrix %d to CSR: ", i);
        sparse_status_t conv_status = mkl_sparse_convert_csr (
            mkl_csc_matrices[i],
            SPARSE_OPERATION_NON_TRANSPOSE,
            &(mkl_csr_matrices[i])
        );
        //switch(conv_status){
            //case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            //case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            //case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            //case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            //case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            //case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            //case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        //}
        //printf("\n");
        /**/
    }
    //printf("MKL sparse matrices created\n");
    //printf("\n");

    //std::vector<int> threads{1, 6, 12, 24, 48};
    //std::vector<int> threads{1, 16, 48};
    //std::vector<int> threads{48, 1, 12};
    std::vector<int> threads{48};

    int iterations = 1;

    for(int i = 0; i < threads.size(); i++){
        //omp_set_num_threads(threads[i]);
        //mkl_set_num_threads(threads[i]);

        omp_set_num_threads(t);
        mkl_set_num_threads(t);

        sparse_matrix_t* mkl_temp = (sparse_matrix_t*) malloc( 1 * sizeof(sparse_matrix_t) );
        double mkl_time = 0;
        matrix_descr desc;
        desc.type = SPARSE_MATRIX_TYPE_GENERAL;
        for(int u = 0; u < k; u++){
            if(mkl_sums[u] != NULL){
                mkl_sparse_destroy(mkl_sums[u]);
            }
        }
        free(mkl_sums);
        mkl_sums = (sparse_matrix_t*) malloc( k * sizeof(sparse_matrix_t) );
        for(int u = 0; u < k; u++){
            mkl_sums[u] = NULL;
        }
        int nIntermediate = k;
        while(nIntermediate > 1){
            //printf("MKL pairwise tree intermediate %d\n", nIntermediate);
            int j = 0;
            int idxf = j * 2 + 0;
            int idxs = idxf;
            if(idxs + 1 < nIntermediate) idxs++;
            while(idxs < nIntermediate){
                if(idxf < idxs){
                    clock.Start();
                    mkl_sparse_d_add(
                        SPARSE_OPERATION_NON_TRANSPOSE, 
                        mkl_csr_matrices[idxf], 
                        1.0, 
                        mkl_csr_matrices[idxs], 
                        mkl_temp
                    );
                    clock.Stop();
                    mkl_time += clock.Seconds();
                    mkl_sparse_destroy(mkl_csr_matrices[idxf]);
                    mkl_sparse_destroy(mkl_csr_matrices[idxs]);
                    mkl_sparse_copy(*mkl_temp, desc, &(mkl_csr_matrices[j]));
                    mkl_sparse_destroy(*mkl_temp);
                }
                else{
                    clock.Start();
                    mkl_sparse_copy(mkl_csr_matrices[idxf], desc, mkl_temp);
                    clock.Stop();
                    //mkl_time += clock.Seconds();
                    mkl_sparse_destroy(mkl_csr_matrices[idxf]);
                    mkl_sparse_copy(*mkl_temp, desc, &(mkl_csr_matrices[j]));
                    mkl_sparse_destroy(*mkl_temp);
                }
                j++;
                idxf = j * 2 + 0;
                idxs = idxf;
                if(idxs + 1 < nIntermediate) idxs++;
            }
            nIntermediate = j;
        }

        //printf("Transposing MKL output: ");
        sparse_matrix_t *mkl_out = (sparse_matrix_t *) malloc( sizeof(sparse_matrix_t) );
        sparse_status_t conv_status = mkl_sparse_convert_csr(
            mkl_csr_matrices[0],
            SPARSE_OPERATION_TRANSPOSE, // Transpose because it will make CSR matrix to be effectively CSC
            mkl_out
        );
        //switch(conv_status){
            //case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            //case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            //case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            //case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            //case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            //case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            //case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        //}
        //printf("\n");

        //printf("Exporting MKL output: ");
        sparse_index_base_t out_indexing;
        MKL_INT out_nrows;
        MKL_INT out_ncols;
        MKL_INT *out_pointerB = NULL;
        MKL_INT *out_pointerE = NULL;
        MKL_INT *out_rows = NULL;
        double *out_values = NULL;
        sparse_status_t export_status = mkl_sparse_d_export_csr (
            *mkl_out,
            &out_indexing,
            &out_nrows,
            &out_ncols,
            &out_pointerB,
            &out_pointerE,
            &out_rows,
            &out_values
        );
        //switch(export_status){
            //case SPARSE_STATUS_SUCCESS: printf("SPARSE_STATUS_SUCCESS"); break;
            //case SPARSE_STATUS_NOT_INITIALIZED: printf("SPARSE_STATUS_NOT_INITIALIZED"); break;
            //case SPARSE_STATUS_ALLOC_FAILED: printf("SPARSE_STATUS_ALLOC_FAILED"); break;
            //case SPARSE_STATUS_INVALID_VALUE: printf("SPARSE_STATUS_INVALID_VALUE"); break;
            //case SPARSE_STATUS_EXECUTION_FAILED: printf("SPARSE_STATUS_EXECUTION_FAILED"); break;
            //case SPARSE_STATUS_INTERNAL_ERROR: printf("SPARSE_STATUS_INTERNAL_ERROR"); break;
            //case SPARSE_STATUS_NOT_SUPPORTED: printf("SPARSE_STATUS_NOT_SUPPORTED"); break;
        //}
        //printf("\n");
        mkl_sparse_destroy(*mkl_out);
        if(type == 0){
            std::cout << "ER" << "," ;
        }
        else if(type == 1){
            std::cout << "RMAT" << "," ;
        }
        else if(type == 2){
            std::cout << "Given" << "," ;
        }
        std::cout << x << "," ;
        std::cout << y << "," ;
        std::cout << d << "," ;
        std::cout << k << "," ;
        std::cout << t << "," ;
        std::cout << "TreeMKL" << ","; 
        std::cout << mkl_time << ",";
        std::cout << total_nnz_in << ",";
        std::cout << out_pointerE[out_ncols-1] << std::endl;
    }

    for (int i = 0; i < k; i++){
       if(mkl_values[i] != NULL) free(mkl_values[i]); 
       if(mkl_rows[i] != NULL) free(mkl_rows[i]); 
       if(mkl_pointerB[i] != NULL) free(mkl_pointerB[i]);
       if(mkl_pointerE[i] != NULL) free(mkl_pointerE[i]);
       //if(mkl_csc_matrices[i] != NULL) mkl_sparse_destroy(mkl_csc_matrices[i]);
       //if(mkl_csr_matrices[i] != NULL) mkl_sparse_destroy(mkl_csr_matrices[i]);
       //if(mkl_sums[i] != NULL) mkl_sparse_destroy(mkl_sums[i]);
    }
    if(mkl_values != NULL) free(mkl_values);
    if(mkl_rows != NULL) free(mkl_rows);
    if(mkl_pointerB != NULL) free(mkl_pointerB);
    if(mkl_pointerE != NULL) free(mkl_pointerE);
    if(mkl_csc_matrices != NULL) free(mkl_csc_matrices);
    if(mkl_csr_matrices != NULL) free(mkl_csr_matrices);
    if(mkl_sums != NULL) free(mkl_sums);

	return 0;

}
