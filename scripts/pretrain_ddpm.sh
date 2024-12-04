#!/bin/bash

python src/ddpm.py \
  --model_dim 64 \
  --channels 1 \
  --dim_mults 1 2 4 8 \
  --flash_attn \
  --image_size 256 \
  --timesteps 1000 \
  --sampling_timesteps 500 \
  --dataset_path "/acfs-home/abh4006/serag_AI_lab/users/abh4006/scripts/ssl_synth/Syn-Med/tmp_ultra_sound_dt" \
  --results_folder "./results" \
  --train_batch_size 8 \
  --train_lr 5e-5 \
  --train_num_steps 300000 \
  --gradient_accumulate_every 2 \
  --ema_decay 0.995 \
  --amp \
  --save_and_sample_every 50000 \
  --convert_image_to "L" \
  --sample_outdir ""
