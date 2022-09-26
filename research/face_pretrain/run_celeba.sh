# PRETRAIN_CHKPT="/home/xing/ResearchGroup/mae/ckpt_vitb16_IN1K_webface4M/msn_vitb16_IN1k_webface4M_10ep.pth.tar"
# IMAGENET_DIR="/home/xing/datasets/CelebA"

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune_celeba.py \
#     --accum_iter 4 \
#     --batch_size 32 \
#     --model vit_base_patch16 \
#     --finetune ${PRETRAIN_CHKPT} \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#     --dist_eval --data_path ${IMAGENET_DIR}

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune_celeba.py \
#     --accum_iter 4 \
#     --batch_size 32 \
#     --model vit_base_patch16 \
#     --finetune ${PRETRAIN_CHKPT} \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
#     --output_dir "ft_output_IN1KWebFace4M_vitb16" \
#     --dist_eval --nb_classes 40 --data_path ${IMAGENET_DIR}

# PRETRAIN_CHKPT="/home/xing/ResearchGroup/mae/ckpt_vits16_dino_ms1m/checkpoint.pth"
# IMAGENET_DIR="/home/xing/datasets/CelebA"
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune_celeba.py \
#     --accum_iter 1 \
#     --batch_size 128 \
#     --model vit_small_patch16 \
#     --finetune ${PRETRAIN_CHKPT} \
#     --epochs 100 \
#     --blr 5e-4 --layer_decay 0.65 \
#     --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
#     --output_dir "ft_output_ms1m_vits16" \
#     --dist_eval --nb_classes 40 --data_path ${IMAGENET_DIR}

PRETRAIN_CHKPT="/home/xing/ResearchGroup/mae/ckpt_dino_public/dino_vitbase16_pretrain.pth"
IMAGENET_DIR="/home/xing/datasets/CelebA"

# for ns in 3255 1627 843 325
#     do for i in {1..3}
#         do
#         OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune_celeba.py \
#             --accum_iter 1 \
#             --batch_size 64 \
#             --model vit_small_patch16 \
#             --finetune ${PRETRAIN_CHKPT} \
#             --num_samples $ns \
#             --epochs 100 \
#             --warmup_epochs 1 \
#             --seed $i \
#             --blr 5e-4 --layer_decay 0.65 \
#             --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
#             --output_dir "ft_celeba_ms1m_syn_vits16" \
#             --dist_eval --nb_classes 40 --data_path ${IMAGENET_DIR}
#         done
#     done

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_finetune_celeba.py \
    --accum_iter 1 \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --num_samples 162770 \
    --epochs 100 \
    --warmup_epochs 1 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.0 --cutmix 0.0 --reprob 0.25 \
    --output_dir "ft_celeba_dino_im1k_vitb16" \
    --dist_eval --nb_classes 40 --data_path ${IMAGENET_DIR}

# PRETRAIN_CHKPT="/home/xing/ResearchGroup/dino/ckpt_vitbase16_ms1m_epochs20/checkpoint.pth"
# IMAGENET_DIR="/home/xing/datasets/CelebA"

# for ns in 3255 1627 843 325
#     do for i in {1..3}
#         do
#         OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 main_linprobe_celeba.py \
#             --accum_iter 1 \
#             --batch_size 64 \
#             --model vit_base_patch16 \
#             --finetune ${PRETRAIN_CHKPT} \
#             --num_samples $ns \
#             --epochs 100 \
#             --warmup_epochs 1 \
#             --blr 5e-4 \
#             --weight_decay 0.05 \
#             --output_dir "linear_celeba_pt_ms1m_vitb16" \
#             --dist_eval --nb_classes 40 --data_path ${IMAGENET_DIR}
#         done
#     done