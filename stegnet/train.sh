# ! /bin/bash
# python3 train_baseline.py \
#     --dataset ../dataset/tiny-imagenet-200 \
#     --output_dir ../ckpt \
#     --loss corr \
    # --load_weights 2022-12-12_22-45-16

python3 train_random.py \
    --dataset ../dataset/tiny-imagenet-200 \
    --output_dir ../ckpt \
#     --loss corr \