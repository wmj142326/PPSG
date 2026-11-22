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
export PROJECT_ROOT=/root/project/PPSG
export PYTHONPATH=$PROJECT_ROOT/src:$PROJECT_ROOT/human_body_prior
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEXT2POSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TEXT2POSE_DIR}" || exit 1

###############################################################################
# CONFIGURATION OF THE EXPERIMENT

###############################################################################
# TEXT INCOMPLETENESS EXPERIMENT
###############################################################################

# 训练文本模式：full | aug
train_text_mode="full"

# 测试文本模式：full | mask | noisy
test_text_mode="full"

# PPSG-Aug训练时，保留完整描述的概率
train_keep_full_prob=0.4

# PPSG-Aug训练时可能采用的Mask比例
train_mask_ratios=(0.1 0.2 0.3 0.5)

# 测试时需要评估的Mask比例
# 可以只设置一个，也可以一次评估多个
test_mask_ratios=(0.1 0.2 0.3 0.5)

# 文本Mask使用的独立随机种子
text_mask_seed=42

# 测试时需要评估的Noisy比例
# 可以只设置一个，也可以一次评估多个
test_noise_ratios=(0.1 0.2 0.3 0.5)

# 文本Noise使用的独立随机种子
text_noise_seed=42

###############################################################################

# default script values
action="train-eval"
run_id_seed=0
sub=""
model_dir_path_override=""
model_path_override=""

print_usage() {
    cat <<EOF
Usage:
  bash generative/script_generative.sh [options]

Basic options:
  -a, --action train|eval|train-eval
  -s, --seed SEED
  -z, --subdir NAME
  --model_dir_path DIR
  --model_path FILE

Text incompleteness options:
  --train_text_mode full|aug
  --test_text_mode full|mask|noisy
  --train_keep_full_prob FLOAT
  --train_mask_ratios R1 [R2 ...]
  --test_mask_ratios R1 [R2 ...]
  --text_mask_seed SEED
  --test_noise_ratios R1 [R2 ...]
  --text_noise_seed SEED

Examples:
  bash generative/script_generative.sh -a train -s 0 -z exp_full --train_text_mode full --test_text_mode full
  bash generative/script_generative.sh -a eval -s 0 -z exp_full --train_text_mode full --test_text_mode mask --test_mask_ratios 0.1 0.2 0.3 0.5 0.7
  bash generative/script_generative.sh -a train-eval -s 0 -z exp_aug --train_text_mode aug --test_text_mode full --train_keep_full_prob 0.4
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--action)
            action="$2"
            shift 2
            ;;
        -s|--seed)
            run_id_seed="$2"
            shift 2
            ;;
        -z|--subdir)
            sub="$2"
            shift 2
            ;;
        --model_dir_path|--model-dir-path)
            model_dir_path_override="$2"
            shift 2
            ;;
        --model_path|--model-path)
            model_path_override="$2"
            shift 2
            ;;
        --train_text_mode|--train-text-mode)
            train_text_mode="$2"
            shift 2
            ;;
        --test_text_mode|--test-text-mode)
            test_text_mode="$2"
            shift 2
            ;;
        --train_keep_full_prob|--train-keep-full-prob)
            train_keep_full_prob="$2"
            shift 2
            ;;
        --train_mask_ratios|--train-mask-ratios)
            shift
            train_mask_ratios=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                ratio_value="${1//,/ }"
                for ratio_part in $ratio_value; do
                    if [[ -n "$ratio_part" ]]; then
                        train_mask_ratios+=("$ratio_part")
                    fi
                done
                shift
            done
            ;;
        --test_mask_ratios|--test-mask-ratios)
            shift
            test_mask_ratios=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                ratio_value="${1//,/ }"
                for ratio_part in $ratio_value; do
                    if [[ -n "$ratio_part" ]]; then
                        test_mask_ratios+=("$ratio_part")
                    fi
                done
                shift
            done
            ;;
        --text_mask_seed|--text-mask-seed)
            text_mask_seed="$2"
            shift 2
            ;;
        --test_noise_ratios|--test-noise-ratios)
            shift
            test_noise_ratios=()
            while [[ $# -gt 0 && "$1" != -* ]]; do
                ratio_value="${1//,/ }"
                for ratio_part in $ratio_value; do
                    if [[ -n "$ratio_part" ]]; then
                        test_noise_ratios+=("$ratio_part")
                    fi
                done
                shift
            done
            ;;
        --text_noise_seed|--text-noise-seed)
            text_noise_seed="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ "${train_text_mode}" != "full" && "${train_text_mode}" != "aug" ]]; then
    echo "Unsupported train_text_mode: ${train_text_mode}"
    exit 1
fi

if [[ "${test_text_mode}" != "full" && "${test_text_mode}" != "mask" && "${test_text_mode}" != "noisy" ]]; then
    echo "Unsupported test_text_mode: ${test_text_mode}"
    exit 1
fi

if [[ "${#train_mask_ratios[@]}" -eq 0 ]]; then
    echo "train_mask_ratios cannot be empty."
    exit 1
fi

if [[ "${#test_mask_ratios[@]}" -eq 0 ]]; then
    echo "test_mask_ratios cannot be empty."
    exit 1
fi

if [[ "${#test_noise_ratios[@]}" -eq 0 ]]; then
    echo "test_noise_ratios cannot be empty."
    exit 1
fi

echo "Text experiment config:"
echo "  train_text_mode=${train_text_mode}"
echo "  test_text_mode=${test_text_mode}"
echo "  train_keep_full_prob=${train_keep_full_prob}"
echo "  train_mask_ratios=${train_mask_ratios[*]}"
echo "  test_mask_ratios=${test_mask_ratios[*]}"
echo "  text_mask_seed=${text_mask_seed}"
echo "  test_noise_ratios=${test_noise_ratios[*]}"
echo "  text_noise_seed=${text_noise_seed}"

result_csv_name="result_test_PVCP2_traintext-${train_text_mode}_testtext-${test_text_mode}_all.csv"
if [[ "${test_text_mode}" == "full" ]]; then
    text_condition_prefix="text-full"
elif [[ "${test_text_mode}" == "noisy" ]]; then
    text_condition_prefix="text-noisy"
else
    text_condition_prefix="text-mask"
fi

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
    --saving_ckpt_step 1
    --log_step 50
    --epochs 3
    --fid 'PVCP2'
    --seed ${run_id_seed}
    --subdir ${sub}
    --train_text_mode "${train_text_mode}"
    --train_keep_full_prob "${train_keep_full_prob}"
    --train_mask_ratios "${train_mask_ratios[@]}"
    --text_mask_seed "${text_mask_seed}"
)

# utils
model_dir_path=$(python option.py "${args[@]}")
if [[ -n "${model_dir_path_override}" ]]; then
    model_dir_path="${model_dir_path_override}"
fi
if [[ -z "${model_dir_path}" ]]; then
    echo "Failed to resolve model_dir_path. Please check option.py and the script arguments."
    exit 1
fi
echo "Model directory: ${model_dir_path}"
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
    if [[ -n "${model_path_override}" ]]; then
        if [[ ! -f "${model_path_override}" ]]; then
            echo "Model checkpoint does not exist: ${model_path_override}"
            exit 1
        fi
        model_dir_path="$(dirname "${model_path_override}")"
        model_paths=("${model_path_override}")
    else
        if [[ ! -d "${model_dir_path}" ]]; then
        echo "Model directory does not exist: ${model_dir_path}"
        echo "Train the model first, or use the same -z/--train_text_mode settings used for training."
        exit 1
        fi

        shopt -s nullglob
        model_paths=("${model_dir_path}"/*.pth)
        shopt -u nullglob

        if [[ "${#model_paths[@]}" -eq 0 ]]; then
            echo "No checkpoint files found in: ${model_dir_path}"
            echo "Expected files like checkpoint_last.pth or checkpoint_99.pth."
            exit 1
        fi
    fi

    for model_path in "${model_paths[@]}"; do
        echo "➡️ Evaluating checkpoint: $model_path"
        if [[ "${test_text_mode}" == "full" ]]; then
            CUDA_VISIBLE_DEVICES=0 python generative/evaluate_generative.py \
                --model_path "$model_path" \
                --split test \
                --test_text_mode full \
                --text_mask_seed "$text_mask_seed" \
                --n_sample 20
        elif [[ "${test_text_mode}" == "mask" ]]; then
            for mask_ratio in "${test_mask_ratios[@]}"; do
                echo "Evaluating sentence-mask ratio: $mask_ratio"
                CUDA_VISIBLE_DEVICES=0 python generative/evaluate_generative.py \
                    --model_path "$model_path" \
                    --split test \
                    --test_text_mode mask \
                    --test_mask_ratio "$mask_ratio" \
                    --text_mask_seed "$text_mask_seed" \
                    --n_sample 20
            done
        elif [[ "${test_text_mode}" == "noisy" ]]; then
            for noise_ratio in "${test_noise_ratios[@]}"; do
                echo "Evaluating attribute-noise ratio: $noise_ratio"
                CUDA_VISIBLE_DEVICES=0 python generative/evaluate_generative.py \
                    --model_path "$model_path" \
                    --split test \
                    --test_text_mode noisy \
                    --test_noise_ratio "$noise_ratio" \
                    --text_noise_seed "$text_noise_seed" \
                    --n_sample 20
            done
        else
            echo "Unsupported test_text_mode: ${test_text_mode}"
            exit 1
        fi
    done

    echo "📌 Now merging result txt files into CSV."
    python generative/merge_results.py \
        --result_dir "$model_dir_path" \
        --split 'test' \
        --dataset 'PVCP2' \
        --output_csv_name "$result_csv_name" \
        --text_condition_prefix "$text_condition_prefix"
fi
