from torch import nn
import torch


class ProbabilityEncoder(nn.Module):
    """Encode probability distributions with statistical features"""
    def __init__(self, input_dim: int, encoding_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        self.prob_encoder = nn.Linear(input_dim, encoding_dim)
        self.moment_encoder = nn.Linear(4, encoding_dim // 4)  # mean, var, skew, kurt
        self.combined = nn.Linear(encoding_dim + encoding_dim // 4, encoding_dim)
        
    def compute_moments(self, prob_dist):
        """Compute statistical moments of probability distribution"""
        # Assume prob_dist is [batch_size, num_bins]
        x = torch.arange(prob_dist.size(-1), dtype=prob_dist.dtype, device=prob_dist.device)
        
        # Mean
        mean = torch.sum(prob_dist * x, dim=-1, keepdim=True)
        
        # Variance
        var = torch.sum(prob_dist * (x - mean)**2, dim=-1, keepdim=True)
        
        # Skewness (simplified)
        skew = torch.sum(prob_dist * (x - mean)**3, dim=-1, keepdim=True) / (var**1.5 + 1e-8)
        
        # Kurtosis (simplified)
        kurt = torch.sum(prob_dist * (x - mean)**4, dim=-1, keepdim=True) / (var**2 + 1e-8)
        
        return torch.cat([mean, var, skew, kurt], dim=-1)
    
    def forward(self, prob_dist):
        # Direct encoding
        prob_encoded = self.prob_encoder(prob_dist)
        
        # Moment encoding
        moments = self.compute_moments(prob_dist)
        moment_encoded = self.moment_encoder(moments)
        
        # Combine
        combined = torch.cat([prob_encoded, moment_encoded], dim=-1)
        return torch.relu(self.combined(combined))