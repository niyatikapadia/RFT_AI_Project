import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from reliability.detection import ChecksumDetector
from reliability.recovery import TemporalRecovery
from reliability.checkpoint import ActivationCheckpoint


class ProtectedAttention(nn.Module):
    """
    Transformer attention with:
    - Checksum-based detection
    - Activation checkpointing
    - Replay-based recovery
    """

    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.detector = ChecksumDetector()
        self.recovery = TemporalRecovery()
        self.checkpoint = ActivationCheckpoint()

    def forward(self, x, inject_fault=False):

        # Save checkpoint
        self.checkpoint.save(x)

        batch_size, seq_len, _ = x.size()

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)

        clean_output = output.clone()

        # Simulate fault
        if inject_fault:
            idx = tuple(torch.randint(0, s, (1,)).item() for s in output.shape)
            output[idx] += 5.0 * torch.randn(1).item()

        # Detection
        if self.detector.detect(clean_output, output):
            print("⚠ Fault detected — replaying layer")
            x_replay = self.checkpoint.load()
            output, attn = self.recovery.replay(self, x_replay)

        return output, attn