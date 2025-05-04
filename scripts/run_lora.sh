data_type="dialog"
load_model='/root/megrez-tmp/model/source/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth'
proj_dir="/root/megrez-tmp/model/3b-$data_type-lora"
data_file="/root/megrez-tmp/dataset/tcm-rwkv/dialog/train"

# n_layer=24
# n_embd=2048
n_layer=32
n_embd=2560

micro_bsz=2
epoch_save=1
epoch_steps=1000
ctx_len=4096

lora_config='{"lora_load":"","lora_r":32,"lora_alpha":32,"lora_dropout":0.0}'

N_NODE=1 # number of nodes
GPU_PER_NODE=1 # number of GPUs per node
DS_BUCKET_MB=2

epoch_count=1000000

python train.py --load_model $load_model \
--wandb "RWKV7-L$n_layer-N$n_embd-3b-$data_type-lora-tuning" \
--num_nodes $N_NODE --devices $GPU_PER_NODE --ds_bucket_mb $DS_BUCKET_MB \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--data_type binidx --dataload pad --loss_mask pad \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count "$epoch_count" --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
--accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x070" \
--peft lora --lora_config $lora_config