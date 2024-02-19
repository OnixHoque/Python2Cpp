#!/bin/bash -l

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

#for ALG in hash-regular-dynamic hash-sliding-dynamic pairwise-tree-dynamic pairwise-serial-dynamic heap-dynamic spa-dynamic mkl-tree mkl-serial
for ALG in spa-dynamic
do
    FILE="rmat.${ALG}.csv"
    #echo "matrix,row-scale,col-scale,d,k,thread,algorithm,total,nnz-in,nnz-out" > $FILE
    #for SCALE in 20 22
    for SCALE in 22
    do
        for K in 4 8 16 32 64 128
        #for K in 128
        do
            for D in 16 32 64 128 256 512 1024 2048
            #for D in 1024
            do
                for T in 48
                do
                    echo ./$ALG $SCALE 10 $D $K 1 $T
                    ./$ALG $SCALE 10 $D $K 1 $T >> $FILE
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
    done
done
