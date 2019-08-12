export CUDA_VISIBLE_DEVICES=3
gpu_fract=.45

dataset="amazon-book_20core"
# model hyper-parameter setting
dim=8
n_hop=1
# n_memory=32
l2_weight=1e-7

# learning parameter setting
lr=5e-3
tolerance=2
early_stop=4
batch_size=1024

topk_eval=True
# command
for n_memory in 2 4 8 16 32
do
emb_name="h${n_hop}_d${dim}_m${n_memory}_sw"
log_name="${dataset}_${emb_name}"

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
    --tolerance $tolerance
    --emb_name $emb_name
    --log_name $log_name
    --topk_eval $topk_eval"
$cmd
done
