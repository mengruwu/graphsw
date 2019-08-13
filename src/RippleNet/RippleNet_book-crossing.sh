export CUDA_VISIBLE_DEVICES=1
gpu_fract=.2

dataset="Book-Crossing"
# model hyper-parameter setting
dim=8
n_hop=1
n_memory=8
l2_weight=1e-6

# learning parameter setting
lr=5e-4
tolerance=0
early_stop=3
stage_early_stop=1
batch_size=1024

# log file setting
emb_name="h${n_hop}_d${dim}_m${n_memory}_sw"
log_name="${dataset}_${emb_name}"

# command
cmd="python3 main.py
    --gpu_fract $gpu_fract
    --dataset $dataset
    --dim $dim
    --n_hop $n_hop
    --n_memory $n_memory
    --l2_weight $l2_weight
    --batch_size $batch_size
    --lr $lr
    --early_stop $early_stop
    --stage_early_stop $stage_early_stop
    --tolerance $tolerance
    --emb_name $emb_name
    --log_name $log_name"
$cmd
