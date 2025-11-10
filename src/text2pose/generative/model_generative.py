import torch
import torch.nn as nn
import numpy as np

from text2pose.data import Tokenizer
from text2pose.encoders import TextEncoder
from text2pose.encoders2 import QwenTextEncoder, PoseEncoder, PoseDecoder



class PPSG(nn.Module):

    def __init__(self, num_neurons=512, latentD=32, text_encoder_name='glovebigru'):
        super(PPSG, self).__init__()
     
        self.latentD = latentD

        # Define pose auto-encoder      
        self.pose_encoder = PoseEncoder(latentD=latentD)
        self.pose_decoder = PoseDecoder(latentD=latentD)

        # Define text encoder
        self.text_encoder_name = text_encoder_name
        if self.text_encoder_name.split("_")[0] in ["Qwen", "qwen"]:
            self.text_encoder = QwenTextEncoder(latentD=latentD, role="generative")
        elif self.text_encoder_name.split("_")[0] in ["glovebigru"]:
            self.text_encoder = TextEncoder(self.text_encoder_name, num_neurons=num_neurons, latentD=latentD, role="generative")
        else:
            raise NotImplementedError(f"Unknown: {self.text_encoder_name}")
        
        # Define learned loss parameters
        self.decsigma_v2v = nn.Parameter( torch.zeros(1) ) # logsigma
        self.decsigma_jts = nn.Parameter( torch.zeros(1) ) # logsigma
        self.decsigma_rot = nn.Parameter( torch.zeros(1) ) # logsigma
        
        # Pose classification head
        self.pose_class_head = nn.Linear(latentD, 4)


    def encode_text(self, captions, caption_lengths):
        return self.text_encoder(captions, caption_lengths)

    def encode_pose(self, pose_body):
        return self.pose_encoder(pose_body)

    def decode(self, z):
        return self.pose_decoder(z)

    def forward(self, pose, captions, caption_lengths):
        t_z = self.encode_text(captions, caption_lengths) #[B,L]->[B,32]
        q_z = self.encode_pose(pose) # [B,52,3]->[B,32]: [torch.Size([32, 24, 3]), torch.Size([32, 24, 9])]
        q_z_sample = q_z.rsample()
        t_z_sample = t_z.rsample()
        ret = {f"{k}_pose":v for k,v in self.decode(q_z_sample).items()}  # ['pose_body', 'pose_body_matrot']_pose
        ret.update({f"{k}_text":v for k,v in self.decode(t_z_sample).items()})  # ['pose_body', 'pose_body_matrot']_text
        ret.update({'q_z': q_z, 't_z': t_z}) # q(z|pose), q(z|text)
        
        # classification logits from latent
        ret['pose_class_logits'] = self.pose_class_head(t_z.mean)
        
        return ret  # dict_keys(['pose_body_pose', 'pose_body_matrot_pose', 'pose_body_text', 'pose_body_matrot_text', 'q_z', 't_z'])
    
    
    def sample_text_nposes(self, captions, caption_lengths, n=1):
        t_z = self.encode_text(captions, caption_lengths)
        z = t_z.sample( [n] ).permute(1,0,2).flatten(0,1)
        decode_results = self.decode(z)
        return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}
        
        
    def sample_str_nposes(self, s, n=1):
        device = self.decsigma_v2v.device
        # no text provided, sample pose directly from latent space
        if len(s)==0:
            z = torch.tensor(np.random.normal(0., 1., size=(n, self.latentD)), dtype=torch.float32, device=device)
            decode_results = self.decode(z)
            return {k: v.view(int(v.shape[0]/n), n, *v.shape[1:]) for k,v in decode_results.items()}
        # otherwise, encode the text to sample a pose conditioned on it
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = Tokenizer(self.text_encoder_name)
        if self.text_encoder_name.split("_")[0] in ["Qwen","qwen"]:
            tokens = self.tokenizer(s)
            return self.sample_text_nposes(tokens, torch.tensor([ len(tokens) ]), n=n)
        elif self.text_encoder_name.split("_")[0] in ["glovebigru"]:
            tokens = self.tokenizer(s).to(device=device)
            return self.sample_text_nposes(tokens.view(1, -1), torch.tensor([ len(tokens) ], dtype=tokens.dtype), n=n)