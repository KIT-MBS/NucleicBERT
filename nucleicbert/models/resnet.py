import torch
import torch.nn as nn
import torch.nn.functional as F

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x, epsilon=1e-10):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True) + epsilon
    a2 = x.sum(-2, keepdims=True) + epsilon
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12 + epsilon)  # Add epsilon to avoid division by very small values
    normalized = x - avg
    return normalized

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ResidualNetwork2D(nn.Module):
    def __init__(self, input_channels, num_residual_blocks, output_channels):
        super(ResidualNetwork2D, self).__init__()
        self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_residual_blocks)]
        )
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch_size, layers, heads, seqlen, _ = x.size()
        attentions = x.view(batch_size, layers * heads, seqlen, seqlen)
        attentions = apc(symmetrize(attentions))
        x = F.relu(self.initial_bn(self.initial_conv(attentions)))
        x = self.residual_blocks(x)
        x = self.final_conv(x)
        x = x.squeeze(1)
        return x
    

class ResidualNetwork2DEMBD(nn.Module):
    def __init__(self, embedding_dim, num_residual_blocks, output_channels):
        super().__init__()
        self.initial_conv = nn.Conv2d(2*embedding_dim, 64, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(64)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_residual_blocks)]
        )
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, embeddings):

        x = embeddings.permute(0, 3, 1, 2)  # [B, 2E, L, L]
        # x = apc(symmetrize(x))
        
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        for block in self.residual_blocks:
            x = self.dropout(block(x))
        
        # Final prediction
        x = self.final_conv(x)
        x = x.squeeze(1)  # [B, L, L]
        
        x = 0.5 * (x + x.transpose(-1, -2))
        
        return x
