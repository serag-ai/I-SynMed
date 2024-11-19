#!/bin/bash

python src/ddpm.py \
  --model_dim 64 \
  --channels 1 \
  --dim_mults 1 2 4 8 \
  --flash_attn True \
  --image_size 256 \
  --timesteps 1000 \
  --sampling_timesteps 500 \
  --dataset_path "/acfs-home/abh4006/serag_AI_lab/shared/synth_data/merged" \
  --results_folder "./results" \
  --train_batch_size 32 \
  --train_lr 8e-5 \
  --train_num_steps 300000 \
  --gradient_accumulate_every 2 \
  --ema_decay 0.995 \
  --amp True \
  --calculate_fid False \
  --save_and_sample_every 300000 \
  --convert_image_to "L" \
  --is_sample True \
  --sample_outdir "./sample_output_dir/" \
  --sample_epochs 1000 \
  --sample_batchsize 32
