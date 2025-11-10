import os
import torch
import torch.nn as nn
import pickle
import roma
import time
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from human_body_prior.models.vposer_model import NormalDistDecoder, VPoser
import torch.nn.functional as F

import text2pose.config as config

################################################################################
## Miscellaneous modules
################################################################################

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)

################################################################################
## Text encoders
################################################################################

class QwenTextEncoder(nn.Module):
    def __init__(self, num_neurons=512, latentD=32, pretrained_model="Qwen/Qwen-1_8B", # Qwen2.5-1.5B; Qwen-1_8B
                 freeze=False, use_lora=True, role="generative"):
        super().__init__()
        # https://huggingface.co/Qwen
        # https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e
        # https://modelscope.cn/models/Qwen/Qwen2.5-3B/files
        print(f'⭐ Text encoder is [{pretrained_model}]')

        pretrained_model = os.path.join(config.PRETRAIN_LLM_MODEL_DIR, pretrained_model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|endoftext|>"
            print(f"⭐ pad_token manually set to: {self.tokenizer.pad_token}")
        
        self.language_model = AutoModelForCausalLM.from_pretrained(pretrained_model, 
                                                            torch_dtype=torch.float16, 
                                                            trust_remote_code=True,
                                                            )
        # print(self.language_model)
                
        # Optional: insert LoRA modules
        if use_lora:
            print("🔧 Applying LoRA...")
            model_name = os.path.basename(pretrained_model).split('-')[0]
            lora_modules=[]
            if model_name=='Qwen2.5': # Qwen2.5
                lora_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"]
            elif model_name=='Qwen': # Qwen-1
                lora_modules += ['attn.c_attn', 'attn.c_proj',
                                 'mlp.w1', 'mlp.w2', 'mlp.c_proj',]
            print(f"⭐ lora_modules: {lora_modules}")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=lora_modules,
                lora_dropout = 0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.language_model = get_peft_model(self.language_model, lora_config)
            self.language_model.enable_input_require_grads() 

        # Freeze parameters if needed
        if freeze and not use_lora:
            print("❄️  Freezing all Qwen parameters")
            for param in self.language_model.parameters():
                param.requires_grad = False
        else:
            print("🔥 Qwen will be fine-tuned")
            self.language_model.gradient_checkpointing_enable()

        self.language_model.eval() if freeze else self.language_model.train()

        self.hidden_dim = self.language_model.config.hidden_size
        print(f"⭐ self.hidden_dim: [{self.hidden_dim}]")
        self.L1 = nn.Linear(self.hidden_dim, num_neurons)  # -->512
        self.norm = nn.LayerNorm(num_neurons)
        self.gelu = nn.GELU()

        # output layers
        if role == "retrieval":
            self.project = L2Norm()
        elif role == "generative":
            self.project = NormalDistDecoder(num_neurons, latentD)
        else:
            raise NotImplementedError(f"Unknown role: {role}")

    def forward(self, captions, caption_lengths=None):
        # captions: List[str]
        inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(self.language_model.device)
        outputs = self.language_model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1] # [B, L, H]

        hidden = last_hidden.mean(dim=1)  # [B, H]  # [B, 2048]
        hidden = hidden.to(self.L1.weight.dtype)
        hidden = self.L1(hidden)  # [B,2048]->[B,512]
        hidden = self.norm(hidden)
        hidden = self.gelu(hidden)

        if isinstance(self.project, L2Norm):
            return self.project(hidden)
        else:
            hidden = hidden.to(self.project.mu.weight.dtype)
            return self.project(hidden)


################################################################################
## Pose encoder / auto-encoder
################################################################################

class Object(object):
    pass

class PoseEncoder(nn.Module):

    def __init__(self, num_neurons=512, num_neurons_mini=32, latentD=512, 
                 role="generative"):
        super(PoseEncoder, self).__init__()
        
        self.num_joints = config.SMPL_NB_INPUT_JOINTS
        self.input_dim = self.num_joints * 3
        
        vposer_params = Object()
        vposer_params.model_params = Object()
        vposer_params.model_params.num_neurons = num_neurons
        vposer_params.model_params.latentD = latentD

        # use VPoser pose encoder architecture...
        vposer = VPoser(vposer_params)
        encoder_layers = list(vposer.encoder_net.children())
        
        # change first layers to have the right data input size
        encoder_layers[1] = nn.BatchNorm1d(self.input_dim)
        encoder_layers[2] = nn.Linear(self.input_dim, num_neurons)
        
        # remove last layer; the last layer.s depend on the task/role
        encoder_layers = encoder_layers[:-1]
        
        # output layers
        if role == "retrieval":
            encoder_layers += [
                nn.Linear(num_neurons, num_neurons_mini),
                nn.ReLU(),
                nn.Linear(num_neurons_mini, latentD),
                L2Norm()]
        elif role == "generative":
            encoder_layers += [ NormalDistDecoder(num_neurons, latentD) ]
        else:
            raise NotImplementedError

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, pose):
        return self.encoder(pose)

class PoseDecoder(nn.Module):

    def __init__(self, num_neurons=512, latentD=32):
        super(PoseDecoder, self).__init__()
        
        self.num_joints = config.SMPL_NB_INPUT_JOINTS

        # use VPoser pose decoder architecture...
        vposer_params = Object()
        vposer_params.model_params = Object()
        vposer_params.model_params.num_neurons = num_neurons
        vposer_params.model_params.latentD = latentD
        vposer = VPoser(vposer_params)
        decoder_layers = list(vposer.decoder_net.children())
        # change one of the final layers to have the right data output size
        decoder_layers[-2] = nn.Linear(num_neurons, self.num_joints * 6)

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, Zin):
        bs = Zin.shape[0]
        prec = self.decoder(Zin)
        return {
            'pose_body': roma.rotmat_to_rotvec(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }    


def count_MB(module):
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total_MB = total_params * 4 / 1024 / 1024
    trainable_MB = trainable_params * 4 / 1024 / 1024
    return total_MB, trainable_MB

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = 'Qwen/Qwen-1_8B'
    encoder = QwenTextEncoder(latentD=32, pretrained_model=model_path, freeze=False, use_lora=True).to(device)
    total_MB, trainable_MB = count_MB(encoder)

    print(total_MB,trainable_MB)
    captions = [
        "He was hit by a car and fell backwards. The man flew through the air and landed on his left side."
    ]

    with torch.no_grad():
        q = encoder(captions)

    # print latent information
    print("Latent distribution:")
    print("  Mean shape:", q.loc.shape)     # (batch, latentD)
    print("  Std shape :", q.scale.shape)   # (batch, latentD)
    print("  Sample     :", q.rsample()[0])  # display the first sample