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

outdir=../datasets/R2R/trained_models/vitbase-finetune_e2e_dst

flag="--root_dir ../datasets
      --output_dir ${outdir}
      --dataset r2r
      --vlnbert ${vlnbert}
      --ob_type ${ob_type}
      
      --world_size ${ngpus}
      --seed ${seed}

      --num_dst_layer 1
      --DST
      --com_weight_from txt_hist_cls
      --norm_la_weight
      --LWeight 1.0
      --AWeight 1.0
      --laspeaker ../datasets/R2R/trained_models/vitbase-finetune_e2e_laspeaker/ckpts/laspeaker/best_val_unseen_loss
      
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
      --lr 1e-5
      --iters 300000
      --log_every 2000
      --batch_size 16
      --optim adamW
      --ml_weight 0.2      
      --feat_dropout 0.4
      --dropout 0.5"

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt


