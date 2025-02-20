import torch
import torch.nn.functional as F
import numpy as np

class Diffusion:
    @staticmethod
    def linear_beta_scheduler(timesteps):
            beta_start = 0.0001
            beta_end = 0.02
            return torch.linspace(beta_start, beta_end, timesteps)

    def __init__(self, timesteps, device):
        self.timesteps = timesteps
        self.device = device
        self.betas = self.linear_beta_scheduler(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, axis = 0).to(device)

        
    def forward_diff(self, x0, t):
        noise = torch.randn_like(x0).to(self.device)
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise, noise
        
    def reverse_diff(self, x, t, noise_pred):
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        return (x - beta_t * noise_pred / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)
