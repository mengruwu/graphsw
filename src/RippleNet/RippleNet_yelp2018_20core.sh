export CUDA_VISIBLE_DEVICES=0

dataset="yelp2018_20core"
# model hyper-parameter setting
dim=8
n_hop=1
n_memory=2
l2_weight=1e-7

# learning parameter setting
lr=5e-3
tolerance=2
early_stop=5

# log file setting
emb_name="h${n_hop}_d${dim}_m${n_memory}_sw"
log_name="${dataset}_${emb_name}"

# command
cmd="python3 main.py
    --dataset $dataset
    --dim $dim
    --n_memory $n_memory
    --l2_weight $l2_weight
    --lr $lr
    --early_stop $early_stop
    --tolerance $tolerance
    --emb_name $emb_name
    --log_name $log_name"
$cmd
