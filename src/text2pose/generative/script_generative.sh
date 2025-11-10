#!/bin/bash

##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################


###############################################################################
# SCRIPT ARGUMENTS
export PROJECT_ROOT=/home/meijun/Projects/PPSG
export PYTHONPATH=$PROJECT_ROOT/src
export TOKENIZERS_PARALLELISM=false

while getopts a:s:z: flag
do
    case "${flag}" in
        a) action=${OPTARG};; # action (train|eval|train-eval)
        s) run_id_seed=${OPTARG};; # (optional) seed value
        z) sub=${OPTARG};; # (optional) subdir
    esac
done

# default values
: ${action="train-eval"}
: ${run_id_seed=0}
: ${sub=""}

###############################################################################
# CONFIGURATION OF THE EXPERIMENT
# global/default configuration
args=(
    --text_encoder_name 'qwen_1_8B_PVCP2'
    --dataset 'PVCP2'
    --latentD 32
    --wloss_v2v 4 --wloss_rot 2 --wloss_jts 2
    --wloss_kld 0.2
    --wloss_cls 0.0001
    --lr 0.0001 --wd 0.0001
    --batch_size 32
    --saving_ckpt_step 5
    --log_step 50
    --epochs 100
    --fid 'PVCP2'
    --seed ${run_id_seed}
    --subdir ${sub}
)

# utils
model_dir_path=$(python option.py "${args[@]}")
# model_path="${model_dir_path}/checkpoint_85.pth" # used for evaluation


###############################################################################
# TRAIN

if [[ "$action" == *"train"* ]]; then
        
    echo "📌 Now training the generative model."
    CUDA_VISIBLE_DEVICES=0 python generative/train_generative.py "${args[@]}"

fi

###############################################################################
# EVAL QUANTITATIVELY

# if [[ "$action" == *"eval"* ]]; then

#     echo "📌 Now computing the metrics."
#     CUDA_VISIBLE_DEVICES=0 python generative/evaluate_generative.py --model_path $model_path --split 'test'

# fi

if [[ "$action" == *"eval"* ]]; then
    echo "📌 Now computing the metrics."
    for model_path in "${model_dir_path}"/*.pth; do
        echo "➡️ Evaluating checkpoint: $model_path"
        CUDA_VISIBLE_DEVICES=0 python generative/evaluate_generative.py --model_path "$model_path" --split 'test'
    done
fi