#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="."

dataset="Book-Crossing"
aggregator="sum"
n_epochs=20
neighbor_sample_size=4
dim=16
n_iter=2
batch_size=2048
l2_weight=2e-5
ls_weight=0.5
lr=8e-4
tolerance=8
early_decrease_lr=2
early_stop=2

# top_k exp
cmd_min="python model/main_top_k.py --log_name book_sw_top_k_${dim} --save_model_name book_sw_top_k_${dim} --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
	  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"
$cmd_min

# by hop exp
cmd_min="python model/main_by_hop.py --log_name book_sw_hop_${dim} --save_model_name book_sw_hop_${dim} --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
	  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"

$cmd_min

# transfer whole parameter
neighbor_sample_size=4
cmd_min="python model/main_by_hop_TA.py --log_name book_sw_hop_TA_${dim} --save_model_name book_sw_hop_TA_${dim} --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
	  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"

$cmd_min
# # parameter iterate

NEIGHBOR_SIZE=(64 32 16 8 4 2)

for NEIGHBOR in ${NEIGHBOR_SIZE[@]}
do

	neighbor_sample_size=$NEIGHBOR

	cmd_min="python model/main.py --log_name book_parameter_fnb --save_model_name book_parameter_fnb --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
		  --dim $dim --n_iter $n_iter --tolerance $tolerance --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"

	$cmd_min

done

# neighbor_sample_size=4

# DIM_SIZE=(128 64 32 16 8 4)

# for DIM_S in ${DIM_SIZE[@]}
# do

# 	dim=$DIM_S

# 	cmd_min="python model/main.py --log_name book_dim_size --save_model_name book_dim_size --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
# 		  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"

# 	$cmd_min

# done

