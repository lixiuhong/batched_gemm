#!/bin/bash

rm -f log
for ((M=128; M<=1024; M=M*2))
do
	for ((K=16; K<=1024; K=K*2))
	do
		cd ../data
		./gen_data $M $M $K
		cd - > /dev/null
		./gemm 4 >> log
		./gemm 8 >> log
		./gemm 16 >> log
		./gemm 32 >> log
		./gemm 64 >> log
		./gemm 128 >> log
		./gemm 256 >> log
	done
done
