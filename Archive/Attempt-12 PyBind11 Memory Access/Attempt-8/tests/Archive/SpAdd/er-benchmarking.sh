#!/bin/bash -l

export OMP_NUM_THREADS=48
export OMP_PLACES=cores

#for ALG in hash-regular-dynamic hash-sliding-dynamic pairwise-tree-dynamic pairwise-serial-dynamic heap-dynamic spa-dynamic mkl-tree mkl-serial
for ALG in spa-dynamic
do
    FILE="er.${ALG}.csv"
    #echo "matrix,row-scale,col-scale,d,k,thread,algorithm,total,nnz-in,nnz-out" > $FILE
    #for RSCALE in 20 22
    for RSCALE in 22
    do
        #for CSCALE in 8 10
        for CSCALE in 10
        do
            for K in 4 8 16 32 64 128
            do
                for D in 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072
                do
                    for T in 48
                    do
                        echo ./$ALG $RSCALE $CSCALE $D $K 0 $T
                        ./$ALG $RSCALE $CSCALE $D $K 0 $T >> $FILE
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
done
