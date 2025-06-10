# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial
from model_utils.classification_head import ClassificationHead
from model_utils.block_exansion import expand_dinov2
from model_utils.lora_adaptation import apply_lora_to_dinov2

import torch
import torch.nn as nn
import copy
import numpy as np
import random

import timm.models.vision_transformer

Dino_embedding_dict = {"s": 384, "b": 768, "l":1024}

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
class Dinov2_Vit(nn.Module):
    def __init__(self, global_pool=False, **kwargs):
        super(Dinov2_Vit, self).__init__()

        self.forward_patches = kwargs['forward_patches']

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.drop_path_rate = kwargs['drop_path_rate']
        
        # Add the original model as a submodule
        self.backbone = kwargs['backbone']

        # Get embedding size
        in_features = self.backbone.blocks[-1].ls1.gamma.size(0)
        if self.forward_patches == 'both':
            in_features = in_features * 2

        # Add additional layers
        if kwargs['block_expansion_positions'] is not None:
            print("Additional layers are added!")
            self.backbone = expand_dinov2(self.backbone, kwargs['block_expansion_positions'], kwargs['block_expansion_path_dropout'])

        if kwargs['lora_adaptation'] == True:
            self.backbone = apply_lora_to_dinov2(
                self.backbone,
                target_blocks=kwargs['lora_adaptation_target_blocks'],
                rank=kwargs['lora_adaptation_rank'],
                alpha=kwargs['lora_adaptation_alpha'],
                adapt_attention=kwargs['lora_adaptation_adapt_attention'],
                adapt_mlp=kwargs['lora_adaptation_adapt_mlp'],
            )

        self.head = ClassificationHead(
            in_features = in_features,
            num_classes = kwargs['num_classes'],
            n_head_layers = kwargs['n_classification_heads'],
            drop_path_rate = self.drop_path_rate
        )
    def block_expansion(self):
        last_block = copy.deepcopy(self.backbone.blocks[-1])

        # Initialize LayerScale parameters (ls1 and ls2) zero
        init_value = 0.0
        dim = last_block.ls1.gamma.size(0)  # Assuming ls1 and ls2 have the same dimension
        last_block.ls1.gamma = nn.Parameter(init_value * torch.ones(dim))
        last_block.ls2.gamma = nn.Parameter(init_value * torch.ones(dim))

        # Add the block to the backbone
        self.backbone.blocks.append(last_block)

    def forward(self, x):

        if self.forward_patches == 'patch':
            x = self.backbone(x, is_training=True)['x_norm_patchtokens']
            # Get the mean of the patch tokens
            x = torch.mean(x, dim=-2)
        elif self.forward_patches == 'cls':
            # Forward pass through the original model
            x = self.backbone(x, is_training=True)['x_norm_clstoken']
        elif self.forward_patches == 'both':
            x1 = self.backbone(x, is_training=True)['x_norm_patchtokens']
            x1 = torch.mean(x1, dim=-2)
            x2 = self.backbone(x, is_training=True)['x_norm_clstoken']
            x = torch.cat((x1, x2), dim=-1)
        # Forward pass through classification head
        x = self.head(x)
        return x


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def dinov2_vit_14(**kwargs):
    model = Dinov2_Vit(**kwargs)
    return model






def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)  # Set the seed for PyTorch
    np.random.seed(seed_value)     # Set the seed for NumPy
    random.seed(seed_value)        # Set the seed for Python's random module
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # If you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




if __name__=='__main__':
    seed_value = 42
    set_seed(seed_value)  # Set a seed value
    model_arch = "dinov2_vitb14"
    model_backbone = torch.hub.load('facebookresearch/dinov2', model_arch)
    model_backbone_normal = torch.hub.load('facebookresearch/dinov2', model_arch)
    model_with_block = Dinov2_Vit(
        num_classes = 5,
        backbone = model_backbone,
        model = model_arch,
        drop_path_rate = 0,
        n_classification_heads = 1,
        block_expansion = [0, 11]
    )
    
    model_normal = Dinov2_Vit(
        num_classes = 5,
        backbone = model_backbone_normal,
        model = model_arch,
        drop_path_rate = 0,
        n_classification_heads = 1,
    )
    print("Model Blockexpansion = %s" % str(model_with_block))
    print("Model Normal = %s" % str(model_normal))
    # Test block expansion
    print("Test block expansion")
    print("Before block expansion:")
    image = torch.randn(1, 3, 224, 224).cuda()
    model_normal.eval().cuda()
    model_with_block.eval().cuda()
    output_normal = model_normal(image)
    output_with_block = model_with_block(image)
    # print(f"Model output normal: {output_normal}")
    # print(f"Model output with block: {output_with_block}")
    # Check if the output is the same
    print(torch.equal(output_normal, output_with_block))
    # difference
    print(f"Difference: {torch.sum(torch.abs(output_normal - output_with_block))}")

    pass

