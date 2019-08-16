#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="."

dataset="MovieLens-1M"
aggregator="sum"
n_epochs=20
neighbor_sample_size=8
dim=16
n_iter=2
batch_size=1024
l2_weight=1e-7
ls_weight=1.0
lr=4e-3
tolerance=4
early_decrease_lr=2
early_stop=2

# top_k exp
cmd_min="python model/main_top_k.py --log_name movie_sw_top_k_${dim} --save_model_name movie_sw_top_k_${dim} --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
	  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"
$cmd_min

# by hop exp
cmd_min="python model/main_by_hop.py --log_name movie_sw_hop_${dim} --save_model_name movie_sw_hop_${dim} --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
	  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"

$cmd_min

# transfer whole parameter
neighbor_sample_size=4
cmd_min="python model/main_by_hop_TA.py --log_name movie_sw_hop_TA_${dim} --save_model_name movie_sw_hop_TA_${dim} --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
	  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"

$cmd_min
# # parameter iterate

NEIGHBOR_SIZE=(64 32 16 8 4 2)

for NEIGHBOR in ${NEIGHBOR_SIZE[@]}
do

	neighbor_sample_size=$NEIGHBOR

	cmd_min="python model/main.py --log_name movie_parameter_fnb --save_model_name movie_parameter_fnb --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
		  --dim $dim --n_iter $n_iter --tolerance $tolerance --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"

	$cmd_min

done

# neighbor_sample_size=4

# DIM_SIZE=(128 64 32 16 8 4)

# for DIM_S in ${DIM_SIZE[@]}
# do

# 	dim=$DIM_S

# 	cmd_min="python model/main.py --log_name movie_dim_size --save_model_name movie_dim_size --dataset $dataset --aggregator $aggregator --n_epochs $n_epochs --neighbor_sample_size $neighbor_sample_size 
# 		  --dim $dim --n_iter $n_iter --batch_size $batch_size --l2_weight $l2_weight --ls_weight $ls_weight --lr $lr --early_decrease_lr $early_decrease_lr --early_stop $early_stop"

# 	$cmd_min

# done

