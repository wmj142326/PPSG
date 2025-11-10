import torch
import torch.nn as nn
import numpy as np

import text2pose.config as config
import text2pose.utils as utils
from text2pose.encoders2 import PoseEncoder

class FID(nn.Module):
    
    def __init__(self, version, model, device=torch.device('cpu')):
        super().__init__()
        self.version = version
        self.device = device
        self.encoder_model = model
        self._load_model()
        
    def sstr(self):
        # return f"FID_{self.version[0]}_seed{self.version[1]}"
        return f"FID"


    def _load_model(self):
        # ckpt = torch.load(config.shortname_2_model_path[self.version[0]].format(seed=self.version[1]), self.device)
        # self.model = PoseEncoder(latentD=ckpt['args'].latentD, role="retrieval")
        # self.model.load_state_dict({k[len('pose_encoder.'):]: v for k,v in ckpt['model'].items() if k.startswith('pose_encoder.encoder.')})
        self.model = self.encoder_model.pose_encoder
        self.model.eval()
        self.model.to(self.device)

    def extract_features(self, batchpose):
        batchpose = batchpose.to(self.device)
        batchpose = batchpose.view(batchpose.size(0),-1)[:,:self.model.input_dim]
        features = self.model(batchpose)
        
        if isinstance(features, torch.distributions.Normal):
            features = features.loc   # or features.rsample()
        return features
    
    def extract_real_features(self, valdataloader):
        real_features = []
        with torch.inference_mode():
            for batches in valdataloader:
                real_features.append(self.extract_features(batches['pose']) )
        self.real_features = torch.cat(real_features, dim=0).cpu().numpy()
        self.realmu = np.mean(self.real_features, axis=0)
        self.realsigma = np.cov(self.real_features, rowvar=False)
        print('FID: extracted real features', self.real_features.shape)
        
    def reset_gen_features(self):
        self.gen_features = []
        
    def add_gen_features(self, batchpose):
        with torch.inference_mode():
            self.gen_features.append( self.extract_features(batchpose) )
        
    def compute(self):
        gen_features = torch.cat(self.gen_features, dim=0).cpu().numpy()
        assert gen_features.shape[0] == self.real_features.shape[0]
        mu = np.mean(gen_features, axis=0)
        sigma = np.cov(gen_features, rowvar=False)
        return utils.calculate_frechet_distance(mu, sigma, self.realmu, self.realsigma)
