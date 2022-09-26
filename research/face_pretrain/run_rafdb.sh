PRETRAIN_CHKPT="/home/xing/ResearchGroup/mae/ckpt_vits16_dino_ms1m_syn/checkpoint-8m.pth"
IMAGENET_DIR="/data/RAF-DB/basic"

# for lr in 1e-3 5e-4 1e-4 5e-5
for lr in 1e-3
    do
    python main_finetune_rafdb.py \
        --accum_iter 1 \
        --batch_size 256 \
        --model vit_small_patch16 \
        --finetune ${PRETRAIN_CHKPT} \
        --epochs 100 \
        --warmup_epochs 1 \
        --blr $lr --layer_decay 0.65 \
        --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
        --output_dir "ft_rafdb_dino_ms1m_8msyn_vits16" \
        --nb_classes 7 --data_path ${IMAGENET_DIR}
    done



# for lr in 1e-3 5e-4 1e-4 5e-5 1e-5
#     do
#     python main_finetune_affecnet.py \
#         --accum_iter 1 \
#         --batch_size 256 \
#         --model vit_small_patch16 \
#         --finetune ${PRETRAIN_CHKPT} \
#         --epochs 10 \
#         --warmup_epochs 0 \
#         --blr $lr --layer_decay 0.65 \
#         --weight_decay 0.05 --drop_path 0.1 --mixup 0.1 --cutmix 0.1 --reprob 0.25 \
#         --output_dir "ft_affecnet_dino_ms1m_8msyn_vits16" \
#         --nb_classes 8 --data_path ${IMAGENET_DIR}
#     done

# PRETRAIN_CHKPT="/home/xing/ResearchGroup/mae/ft_affecnet_dino_ms1m_8msyn_vits16/checkpoint-8.pth"
# IMAGENET_DIR="/data/AfectNet"
# python main_finetune_affecnet.py --eval --resume ${PRETRAIN_CHKPT} --model vit_small_patch16 --batch_size 256 --data_path ${IMAGENET_DIR} \
# --nb_classes 8 \
# --output_dir "ft_affecnet_dino_ms1m_8msyn_vits16"