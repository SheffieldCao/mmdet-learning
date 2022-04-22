# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

import timm
from timm.models.vision_transformer import Block, init_weights_vit_timm, _load_weights
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.helpers import checkpoint_seq

from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict

from ...utils import get_root_logger
from ..builder import BACKBONES
# from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw, pvt_convert


@BACKBONES.register_module()
class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    
    Modified from timm.models.vision_transformer.py implementation.
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, global_pool='token', embed_dim=768, depth=12, 
            num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, drop_rate=0., 
            attn_drop_rate=0., drop_path_rate=0., init_values=None,embed_layer=PatchEmbed, 
            norm_layer=None, act_layer=None, block_fn=Block, init_cfg=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
            block_fn: (nn.Module): ViT block
            init_cfg: (dict): weight initialization config
                default: None
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False

        # generate hw_shape
        self.hw_shape = [img_size[0]//patch_size, img_size[1]//patch_size] if isinstance(img_size, tuple) else [img_size//patch_size, img_size//patch_size]

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        # self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        # final_chs = self.representation_size if self.representation_size else self.embed_dim
        # self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

        self.init_cfg = init_cfg
        self.init_weights()

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self):
        logger = get_root_logger()
        # init pos_embed
        trunc_normal_(self.pos_embed, std=.02)

        assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                f'specify `Pretrained` in ' \
                                                f'`init_cfg` in ' \
                                                f'{self.__class__.__name__} '
        checkpoint = _load_checkpoint(
            self.init_cfg.checkpoint, logger=logger, map_location='cpu')
        logger.warn(f'Load pre-trained model for '
                    f'{self.__class__.__name__} from original repo')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        load_state_dict(self, state_dict, strict=False, logger=logger)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    # def forward_head(self, x, pre_logits: bool = False):
    #     x = self.fc_norm(x)
    #     x = self.pre_logits(x)
    #     return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x

@BACKBONES.register_module()
class ViT(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(ViT, self).__init__(**kwargs)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # x = self.norm(x)
        x = x[:,1:,:]
        
        return x
    
    def _cvt_nld2nchw(self, x):
        H, W = self.hw_shape
        assert len(x.shape) == 3
        B, L, C = x.shape
        assert L == H * W, 'The seq_len does not match H, W'
        return x.transpose(1, 2).reshape(B, C, H, W).contiguous()

    def forward(self, x):
        x = self.forward_features(x)
        x = self._cvt_nld2nchw(x)
        return (x,)