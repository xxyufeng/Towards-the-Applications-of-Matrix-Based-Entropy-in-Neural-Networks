import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings + class token + positional embedding
    Input: (B, C, H, W)
    Output: (B, N+1, D) where N = (H*W)/(P*P), D = embedding dimension
    """
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: linear layer (implemented via Conv2d for efficiency)
        self.patch_embed = nn.Conv2d(
            in_channels=in_ch, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size  # No overlap between patches
        )
        
        # Class token: (1, 1, D) -> expand to (B, 1, D) in forward
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding: (1, N+1, D) -> learnable
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        B = x.shape[0]  # Batch size
        
        # Patch embedding (B, C, H, W) -> (B, D, N^(1/2), N^(1/2))
        x = self.patch_embed(x)  # (B, 256, 8, 8) for img_size=32, patch_size=4
        
        # Flatten patches (B, D, N^(1/2), N^(1/2)) -> (B, D, N)
        x = x.flatten(2)  # (B, 256, 64)
        
        # Transpose to (B, N, D)
        x = x.transpose(1, 2)  # (B, 64, 256)
        
        # Add class token (B, 1, D)
        class_token = self.class_token.expand(B, -1, -1)  # (B, 1, 256)
        x = torch.cat([class_token, x], dim=1)  # (B, 65, 256)
        
        # Add positional embedding
        x = x + self.pos_embed  # (B, 65, 256)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) module
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head
        
        # Ensure embed_dim is divisible by num_heads
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num heads"
        
        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D = x.shape  # (B, 65, 256)
        
        # Compute Q, K, V (B, N, D)
        q = self.q_proj(x)  # (B, 65, 256)
        k = self.k_proj(x)  # (B, 65, 256)
        v = self.v_proj(x)  # (B, 65, 256)
        
        # Split into multiple heads (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 8, 65, 32)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 8, 65, 32)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 8, 65, 32)
        
        # Compute attention scores (B, num_heads, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, 8, 65, 65)
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # Scale
        
        # Softmax to get attention weights (B, num_heads, N, N)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, 8, 65, 65)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum of V (B, num_heads, N, head_dim)
        attn_output = torch.matmul(attn_weights, v)  # (B, 8, 65, 32)
        
        # Concatenate heads (B, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, 65, 8, 32)
        attn_output = attn_output.view(B, N, D)  # (B, 65, 256)
        
        # Linear projection
        output = self.out_proj(attn_output)  # (B, 65, 256)
        output = self.dropout(output)
        
        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Transformer encoder
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, embed_dim=256, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)  # Expand to 4x dimension
        self.fc2 = nn.Linear(mlp_dim, embed_dim)  # Project back
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()  # Activation function
        
    def forward(self, x):
        x = self.fc1(x)  # (B, 65, 256) -> (B, 65, 1024)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 65, 1024) -> (B, 65, 256)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single layer of Transformer encoder
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=1024, dropout=0.1):
        super().__init__()
        # Layer normalization before MHSA
        self.ln1 = nn.LayerNorm(embed_dim)
        # Multi-Head Self-Attention
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        # Layer normalization before MLP
        self.ln2 = nn.LayerNorm(embed_dim)
        # MLP block
        self.mlp = MLP(embed_dim, mlp_dim, dropout)
        
    def forward(self, x):
        # Residual connection + MHSA
        x = x + self.mhsa(self.ln1(x))  # (B, 65, 256)
        # Residual connection + MLP
        x = x + self.mlp(self.ln2(x))  # (B, 65, 256)
        return x


class VisionTransformer(nn.Module):
    """
    Full Vision Transformer model for image classification
    Input: (B, C, H, W)
    Output: (B, num_classes)
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_ch=3,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        mlp_dim=1024,
        num_classes=10,
        dropout=0.1,
        n_iterations=1000
    ):
        super().__init__()
        # Patch embedding + class token + positional embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_ch, embed_dim)
        
        # Transformer encoder (stack multiple layers)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for encoder output
        self.ln = nn.LayerNorm(embed_dim)
        
        # Classification head (linear layer)
        self.classifier = nn.Linear(embed_dim, num_classes)

        #sigma
        self.sigmas = torch.zeros((4, n_iterations)).cuda()
        self.softmax = nn.Softmax(dim=1)
        self.MI = torch.zeros((n_iterations, 3, 2)).cuda()  # To store MI values
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize model weights (improve training stability)"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        # Patch embedding + positional encoding (B, C, H, W) -> (B, N+1, D)
        x = self.patch_embed(x)  # (B, 65, 256)
        
        # Pass through Transformer encoder layers
        for layer in self.encoder_layers[:-1]:
            x = layer(x)  # (B, 65, 256)

        layer_FMSA = x.view(x.size(0), -1)  # Save for MI analysis # (B, 65*256)
        x = self.encoder_layers[-1](x)
        
        # Layer normalization
        x = self.ln(x)  # (B, 65, 256)
        
        # Extract class token feature (B, D)
        class_token_feature = x[:, 0, :]  # (B, 256)
        layer_CLS = class_token_feature  # Save for MI analysis # (B, 256)
        
        # Classification head (B, num_classes)
        logits = self.classifier(class_token_feature)  # (B, 10)
        
        return [logits, layer_CLS, layer_FMSA]
    
    def dist_mat(self, x):
        '''Calculate pairwise Euclidean distance matrix'''
        if len(x.size()) == 4:
            x = x.view(x.size()[0], -1) #transform for input images

        assert len(x.shape) == 2, "Input must be 2D tensor" # (B, D)

        dist = torch.norm(x[:, None] - x, dim=2)
        return dist

    def entropy(self, *args):
        '''
        Calculate matrix-based Renyi's entropy
        args: list of kernel matrices
        if len(args) == 1: H(X)
        if len(args) == 2: J(X,Y)
        '''
        for idx, val in enumerate(args):
            if idx == 0:
                k = val.clone()
            else:
                k *= val

        k /= k.trace()

        eigv = torch.linalg.eigh(k)[0].abs()

        return -(eigv*(eigv.log2())).sum()

    def kernel_mat(self, x, k_x, k_y, sigma=None, epoch=None, idx=None):

        d = self.dist_mat(x)
        if sigma is None:
          if epoch > 40:
            sigma_vals = torch.linspace(0.1, 10*d.mean().item(), 50).cuda()
          else:
            sigma_vals = torch.linspace(0.1, 10*d.mean().item(), 75).cuda()
          L = []
          for sig in sigma_vals:
            k_l = torch.exp(-d ** 2 / (sig ** 2)) / d.size(0)
            L.append(self.kernel_loss(k_x, k_y, k_l, idx))

          if epoch == 0:
            self.sigmas[idx+1, epoch] = sigma_vals[L.index(max(L))]
          else:
            self.sigmas[idx+1, epoch] = 0.9*self.sigmas[idx+1, epoch-1] + 0.1*sigma_vals[L.index(max(L))]

          sigma = self.sigmas[idx+1, epoch]

        return torch.exp(-d ** 2 / (sigma ** 2))


    def kernel_loss(self, k_x, k_y, k_l, idx):

        b = 1.0
        beta = [b, b, b]

        L = torch.norm(k_l)
        Y = torch.norm(k_y) ** beta[idx]
        X = torch.norm(k_x) ** (1-beta[idx])

        LY = torch.trace(torch.matmul(k_l, k_y))**beta[idx]
        LX = torch.trace(torch.matmul(k_l, k_x))**(1-beta[idx])

        return 2*torch.log2((LY*LX)/(L*Y*X))


    def compute_mi(self, x, y, model, current_iteration):

        model.eval()

        data = self.forward(x)
        data.reverse()
        data[-1] = self.softmax(data[-1])
        data.insert(0, x)
        one_hot = F.one_hot(y, num_classes=self.classifier.out_features).float()
        data.append(one_hot.cuda())

        k_x = self.kernel_mat(data[0], [], [], sigma=torch.tensor(8.0).cuda())
        k_y = self.kernel_mat(data[-1], [], [], sigma=torch.tensor(0.1).cuda())

        k_list = [k_x]
        for idx_l, val in enumerate(data[1:-1]):
          k_list.append(self.kernel_mat(val.reshape(data[0].size(0), -1), k_x, k_y, epoch=current_iteration, idx=idx_l))
        k_list.append(k_y)

        e_list = [self.entropy(i) for i in k_list]
        j_XT = [self.entropy(k_list[0], k_i) for k_i in k_list[1:-1]]
        j_TY = [self.entropy(k_i, k_list[-1]) for k_i in k_list[1:-1]]

        for idx_mi, val_mi in enumerate(e_list[1:-1]):
            self.MI[current_iteration, idx_mi, 0] = e_list[0]+val_mi-j_XT[idx_mi]
            self.MI[current_iteration, idx_mi, 1] = e_list[-1]+val_mi-j_TY[idx_mi]

        return
