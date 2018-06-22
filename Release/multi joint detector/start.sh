#!/bin/bash

log_dir=logs

if [ ! -d "$log_dir" ]
then
	mkdir $log_dir
fi

timestamp=`date "+%d.%m.%Y-%H:%M:%S.%3N"`
log_file="$log_dir/$timestamp.log"

if [[ $# -ge 1 ]]
then
	exec=$1
else
	exec=main.py
fi

install_dir=~/anaconda3/bin

gpu_idx=2

echo "executing $exec on GPU nr $gpu_idx" | tee $log_file

CUDA_VISIBLE_DEVICES=$gpu_idx $install_dir/python -u $exec 2>&1 | tee -a $log_file
