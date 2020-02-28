#!/bin/bash

rm -f log
for ((thres=65536; thres<=102400000; thres=thres*2))
do
	./gemm 16 $thres >> log
done
for ((thres=65536; thres<=102400000; thres=thres*2))
do
	./gemm 32 $thres >> log
done
for ((thres=65536; thres<=102400000; thres=thres*2))
do
	./gemm 64 $thres >> log
done
for ((thres=65536; thres<=102400000; thres=thres*2))
do
	./gemm 128 $thres >> log
done
for ((thres=65536; thres<=102400000; thres=thres*2))
do
	./gemm 256 $thres >> log
done
