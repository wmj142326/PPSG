##############################################################
## text2pose                                                ##
## Copyright (c) 2022-present                               ##
## Institut de Robotica i Informatica Industrial, CSIC-UPC  ##
## Naver Corporation                                        ##
## CC BY-NC-SA 4.0                                          ##
##############################################################

# This file serves to store global config parameters and paths
# (those may change depending on the user, provided data, trained models...)

# default
import os,sys
MAIN_DIR = os.path.realpath(__file__) # Absolute path of current script
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(MAIN_DIR))) # Up 3 levels of directory

################################################################################
# Output dir for experiments
################################################################################

GENERAL_EXP_OUTPUT_DIR = MAIN_DIR + '/experiments' # where models will be saved (along with logs, generated poses...)
PRETRAIN_LLM_MODEL_DIR = MAIN_DIR + '/pretrainllm' # LLM models will be saved

################################################################################
# Data
################################################################################

POSESCRIPT_LOCATION = MAIN_DIR + '/data/PoseScript/posescript_release' # where PoseScript is located (vocabulary files will be generated into this directory).
PVCP2_LOCATION = MAIN_DIR + '/data/PVCP2' # where PVCP2 is located (vocabulary files will be generated into this directory).

### SMPLH pose config
SMPLH_BODY_MODEL_PATH = MAIN_DIR + '/data/smplh_amass_body_models'  # where SMPL-H body models are located.
SMPLH_NEUTRAL_BM = f'{SMPLH_BODY_MODEL_PATH}/neutral/model.npz'
n_betas = 16
NB_INPUT_JOINTS = 52

### SMPL pose config
SMPL_BODY_MODEL_PATH = MAIN_DIR + '/data/smpl_body_models'  # where SMPL body models are located.
SMPL_NEUTRAL_BM = f'{SMPL_BODY_MODEL_PATH}/SMPL_N_model.npz' # https://huggingface.co/lithiumice/models_hub/tree/main
SMPL_n_betas = 10
SMPL_NB_INPUT_JOINTS = 24


### pose data
AMASS_FILE_LOCATION = MAIN_DIR + '/data/AMASS/smplh/'  # where AMASS is located.
supported_datasets = {"AMASS":AMASS_FILE_LOCATION}

BABEL_LOCATION = MAIN_DIR + '/data/BABEL/babel_v1.0_release'  # where BABEL is located.

# saved in the directory of the corresponding model
generated_pose_path = '%s/generated_poses/posescript_version_{data_version}_split_{split}_gensamples.pth' # %s is for the model directory (obtainable with shortname_2_model_path)


### text data
MAX_TOKENS = 500 # defined here because it depends on the provided data (only affects the glovebigru configuration)

vocab_files = {
    # IMPORTANT: do not use "_" symbols in the keys of this dictionary
    "vocA1H1": "vocab3893.pkl",
    "vocPVCP2": "vocabpvcp2.pkl",
}

caption_files = {
    "posescript-A1": [f"automatic_{x}.json" for x in ["A", "B", "C", "D", "E", "F"]],
    "posescript-H1": "human3893.json",
}


# data cache
cache_file_path = MAIN_DIR + '/dataset_cache/PoseScript_version_{data_version}_split_{split}_tokenizer_{tokenizer}.pkl'


################################################################################
# Model cache
################################################################################

GLOVE_DIR = MAIN_DIR + '/data/glove' # or None, where glove.840B.300d.txt is located (unzip the downloaded archive).


################################################################################
# Shortnames to checkpoint paths
################################################################################
# Shortnames are used to refer to:
# - pretrained models
# - models that generated pose files
# - retrieval models to serve for fid

# shortnames for models are expected to be the same accross seed values;
# model paths should contain a specific seed_value field instead of the actual seed value
normalize_model_path = lambda model_path, seed_value: "/".join(model_path.split("/")[:-2]) + f"/seed{seed_value}/"+ model_path.split("/")[-1]

################################################################################
# Evaluation
################################################################################

# NOTE: models used to compute the fid should be specified in `shortname_2_model_path`

k_recall_values = [1, 5, 10]