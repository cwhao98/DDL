#!/bin/bash

nvidia-smi

export PYTHONPATH=$PYTHONPATH:`pwd`


ob_type=pano
feedback=sample

features=vitbase_r2rfte2e
# features=vitbase
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/R2R/trained_models/vitbase-finetune_e2e_laspeaker

flag="--root_dir ../datasets
      --output_dir ${outdir}
      --dataset r2r
      --vlnbert ${vlnbert}
      --ob_type ${ob_type}
      
      --world_size ${ngpus}
      --seed ${seed}

      --train laspeaker
      
      --num_l_layers 9
      --num_x_layers 4
      --hist_enc_pano
      --hist_pano_num_layers 2
      --fix_lang_embedding
      --fix_hist_embedding
      --features ${features}
      --feedback ${feedback}
      --max_action_len 15
      --max_instr_len 60
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --lr 1e-4
      --batch_size 16
      --optim adamW
      --ml_weight 0.2      
      --feat_dropout 0.4
      --dropout 0.6"

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag

