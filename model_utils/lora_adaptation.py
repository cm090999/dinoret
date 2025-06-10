# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import copy
from typing import List, Optional


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for linear transformations."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_lora_weights: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize weights
        if init_lora_weights:
            self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA weights following the standard approach."""
        # Initialize A with kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA adapter."""
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        enable_lora: bool = True,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.enable_lora = enable_lora
        
        # Freeze original parameters
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        # Add LoRA adapter
        if enable_lora:
            self.lora_adapter = LoRALayer(
                in_features=original_linear.in_features,
                out_features=original_linear.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
        else:
            self.lora_adapter = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original weights with LoRA adaptation."""
        result = self.original_linear(x)
        if self.enable_lora and self.lora_adapter is not None:
            result = result + self.lora_adapter(x)
        return result


def apply_lora_to_attention(attn_module, rank: int = 4, alpha: float = 16.0, dropout: float = 0.0, target_modules: List[str] = None):
    """Apply LoRA to attention module (QKV and projection layers)."""
    if target_modules is None:
        target_modules = ["qkv", "proj"]
    
    for name in target_modules:
        if hasattr(attn_module, name):
            original_layer = getattr(attn_module, name)
            if isinstance(original_layer, nn.Linear):
                lora_layer = LoRALinear(
                    original_linear=original_layer,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(attn_module, name, lora_layer)


def apply_lora_to_mlp(mlp_module, rank: int = 4, alpha: float = 16.0, dropout: float = 0.0, target_modules: List[str] = None):
    """Apply LoRA to MLP module (fc1 and fc2 layers)."""
    if target_modules is None:
        target_modules = ["fc1", "fc2"]
    
    for name in target_modules:
        if hasattr(mlp_module, name):
            original_layer = getattr(mlp_module, name)
            if isinstance(original_layer, nn.Linear):
                lora_layer = LoRALinear(
                    original_linear=original_layer,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(mlp_module, name, lora_layer)


def apply_lora_to_block(block, rank: int = 4, alpha: float = 16.0, dropout: float = 0.0, adapt_attention: bool = True, adapt_mlp: bool = True):
    """Apply LoRA to a transformer block."""
    if adapt_attention and hasattr(block, 'attn'):
        apply_lora_to_attention(block.attn, rank=rank, alpha=alpha, dropout=dropout)
    
    if adapt_mlp and hasattr(block, 'mlp'):
        apply_lora_to_mlp(block.mlp, rank=rank, alpha=alpha, dropout=dropout)


def freeze_non_lora_parameters(model):
    """Freeze all parameters except LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False


def apply_lora_to_dinov2(
    backbone,
    target_blocks: List[int] = None,
    rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.0,
    adapt_attention: bool = True,
    adapt_mlp: bool = True,
    freeze_non_lora: bool = True,
):
    """Apply LoRA adaptation to specific blocks of a DINOv2 backbone.
    
    Args:
        backbone: The DINOv2 backbone model
        target_blocks: List of block indices to apply LoRA to. If None, applies to all blocks.
        rank: Rank of LoRA adaptation
        alpha: LoRA alpha parameter for scaling
        dropout: Dropout rate for LoRA layers
        adapt_attention: Whether to adapt attention layers
        adapt_mlp: Whether to adapt MLP layers
    
    Returns:
        Modified backbone with LoRA adaptation applied
    """
    if target_blocks is None:
        target_blocks = list(range(len(backbone.blocks)))
    
    adapted_backbone = copy.deepcopy(backbone)
    
    for block_idx in target_blocks:
        if block_idx >= len(adapted_backbone.blocks):
            continue
            
        apply_lora_to_block(
            adapted_backbone.blocks[block_idx],
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            adapt_attention=adapt_attention,
            adapt_mlp=adapt_mlp,
        )
    
    # Freeze all non-LoRA parameters
    if freeze_non_lora:
        freeze_non_lora_parameters(adapted_backbone)
    
    return adapted_backbone


def get_lora_parameters(model):
    """Get all LoRA parameters from the model."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def get_lora_state_dict(model):
    """Get state dict containing only LoRA parameters."""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param
    return lora_state_dict


def enable_lora_adapters(model, enable: bool = True):
    """Enable or disable LoRA adapters in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enable_lora = enable


def print_lora_info(model):
    """Print information about LoRA adapters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"LoRA parameters ratio: {lora_params/total_params:.4f}")
    print(f"Trainable parameters ratio: {trainable_params/total_params:.4f}")


def load_lora_state_dict(model, lora_state_dict):
    """Load LoRA state dict into the model."""
    model_state_dict = model.state_dict()
    
    # Filter to only LoRA parameters that exist in the model
    filtered_lora_state_dict = {}
    for name, param in lora_state_dict.items():
        if name in model_state_dict and 'lora_' in name:
            filtered_lora_state_dict[name] = param
    
    # Load the filtered state dict
    model.load_state_dict(filtered_lora_state_dict, strict=False)
    print(f"Loaded {len(filtered_lora_state_dict)} LoRA parameters")


def save_lora_checkpoint(model, filepath):
    """Save only LoRA parameters to a checkpoint file."""
    lora_state_dict = get_lora_state_dict(model)
    torch.save(lora_state_dict, filepath)
    print(f"Saved LoRA checkpoint with {len(lora_state_dict)} parameters to {filepath}")
