
export CUDA_VISIBLE_DEVICES=3

dataset="Book-Crossing"
dim=4
n_hop=1
n_memory=8
kge_weight=0.01
l2_weight=1e-5
lr=2e-3
batch_size=1024
n_epoch=50
item_update_mode="plus_transform"
using_all_hops=True
early_stop=4
tolerance=0

emb_name="sw_m_${n_memory}_d_${dim}"

cmd="python3 main.py
    --dataset $dataset
    --dim $dim
    --n_hop $n_hop
    --kge_weight $kge_weight
    --l2_weight $l2_weight
    --lr $lr
    --batch_size $batch_size
    --n_epoch $n_epoch
    --item_update_mode $item_update_mode
    --using_all_hops $using_all_hops 
    --early_stop $early_stop
    --tolerance $tolerance
    --n_memory $n_memory
    --emb_name $emb_name
    --log_name ${dataset}_${emb_name}"
$cmd
######################################################
for m in 16
do
    pretrained_emb_name="step_by_step_best__m_${m}_d_${dim}"
    cmd="python3 main_tr.py --mode $mode --log_name ${dataset}_${pretrained_emb_name} --save_model_name ${dataset}_${pretrained_emb_name} --dataset $dataset --dim $dim --n_hop $n_hop --kge_weight $kge_weight --l2_weight $l2_weight --lr $lr
        --batch_size $batch_size --n_epoch $n_epoch --n_memory $m --item_update_mode $item_update_mode --using_all_hops $using_all_hops 
        --beta $beta --early_decrease_lr $early_decrease_lr --early_stop $early_stop --tolerance $tolerance --pretrained_emb_name $pretrained_emb_name"
    $cmd
done
######################################################
export CUDA_VISIBLE_DEVICES=3

dataset="amazon-book"
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