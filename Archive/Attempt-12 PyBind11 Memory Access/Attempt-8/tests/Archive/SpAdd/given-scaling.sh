#!/bin/bash -l

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

RSCALE=22 # Would not matter for given matrices
CSCALE=10 # Would not matter for given matrices
K=64
FILE="given-scaling.csv"
PREFIX="/N/u2/t/taufique/Data/r0_s"
#echo "matrix,row-scale,col-scale,d,k,thread,algorithm,total,nnz-in,nnz-out" > $FILE

#for ALG in hash-regular-dynamic hash-sliding-dynamic pairwise-tree-dynamic pairwise-serial-dynamic heap-dynamic spa-dynamic mkl-tree mkl-serial
for ALG in spa-dynamic
do
    for D in 12345
    do
        for T in 48 24 12 1
        do
            echo ./$ALG $RSCALE $CSCALE $D $K 2 $T $PREFIX
            ./$ALG $RSCALE $CSCALE $D $K 2 $T $PREFIX >> $FILE
            if [ $? -eq 0 ]
            then
                echo OK
            else
                echo FAIL $?
            fi
            echo ---
        done
    done
done
