#!/bin/bash

python3 src/pretrain_dino_vit16.py \
  --output_dim 2048 \
  --learning_rate 0.001 \
  --dataset_path "/acfs-home/abh4006/serag_AI_lab/users/abh4006/scripts/ssl_synth/Syn-Med/merged_synthetic_ultrasound" \
  --epochs 500 \
  --evaluation_epochs 50 \
  --output_dir "/acfs-home/abh4006/serag_AI_lab/users/abh4006/scripts/ssl_synth/Syn-Med/ultrasound_dino_results/dino_lr_1e3/" \
  --batch_size 32