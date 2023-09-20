'''
 * Based on vit from blip code base
 * https://github.com/salesforce/BLIP
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

from timm.models.vision_transformer import _cfg, PatchEmbed, resize_pos_embed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False, prompt=None, prompt_type="tuning"):
        B, N, C = x.shape
        # B is number of batch size
        # N is number of feature vectors (number of patches sub images + 1)
        # C is embedding dimension
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv shape == (3, B, self.num_heads, N, C // self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]   # shape of each == (B, self.num_heads, N, C // self.num_heads)
        prompt_length = 0
        if prompt is not None:
            if prompt_type == "prefix":
                pk, pv = prompt
                pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                k = torch.cat((pk,k), dim=2) # shape == (B, self.num_heads, N + self.top_k * i, C // self.num_heads)
                v = torch.cat((pv,v), dim=2) # shape == (B, self.num_heads, N + self.top_k * i, C // self.num_heads)
            elif prompt_type == "tuning":
                prompt_length = prompt.shape[1]
                prompt = prompt.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                k = torch.cat((prompt, k), dim=2) # shape == (B, self.num_heads, N + Lp, C // self.num_heads)
                v = torch.cat((prompt, v), dim=2) # shape == (B, self.num_heads, N + Lp, C // self.num_heads)
                q = torch.cat((prompt, q), dim=2) # shape == (B, self.num_heads, N + Lp, C // self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if prefix:
        # q @ k.transpose(-2, -1) having shape == (B, self.num_heads, N, N + self.top_k * i)
        # elif tuning:
        # q @ k.transpose(-2, -1) having shape == (B, self.num_heads, N + Lp, N + Lp)
        attn = attn.softmax(dim=-1) # shape == (B, self.num_heads, N, N + self.top_k * i)
        # if prefix: attn.shape == (B, self.num_heads, N, N + self.top_k * i)
        # elif tuning: attn.shape == (B, self.num_heads, N + Lp, N + Lp)

        attn = self.attn_drop(attn)

        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        
        # print(attn.shape, v.shape)
        # attn.shape == ([64, 12, 197, 197]); v.shape == ([64, 12, 197, 64])
        if prompt_type == "prefix":
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        elif prompt_type == "tuning":
            x = (attn @ v).transpose(1, 2).reshape(B, N + prompt_length, C)
            # x = x[:, prompt_length:, :]
            # x = torch.cat((x[:,0,:], x[:,(prompt_length+1):,:]), dim=1) # cut down indices where prompt is located
        # if prefix
        # shape(attn @ v) == (B, self.num_heads, N, N + self.top_k * i) @ ..
        # ..(B, self.num_heads, N + self.top_k * i, C // self.num_heads)
        # == (B, self.num_heads, N, C // self.num_heads)
        # elif tuning:
        # shape(attn @ v) == (B, self.num_heads, N + Lp, N + Lp) @ (B, self.num_heads, N + Lp, C // self.num_heads)
        # == (B, self.num_heads, N + Lp, C // self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        if prompt_type == "tuning":
            x = x[:, prompt_length:, :]

        assert x.shape == (B, N, C), "x.shape != (B, N, C)"
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, register_hook=False, prompt=None):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        # there is unique class_token, meaning class_token is shared among instances
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # class token, corresponding to first vector embedding
        # positional_embedding, used to adding with flatten_patches vector
        # again, there is only one positional embedding, and in posisional embedding, there are num_patches+1 position to be embedded
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) # position embedding
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, block_id_for_register_hook=-1, prompt=None, q=None, train=False, task_id=None, possible_task_id=None):
        B = x.shape[0] # number of instances in a batch
        patch_x = self.patch_embed(x) # output of Embedding Patch layer

        # class token, basically this line of code is used for copycatting class token to each instance in the batch
        # there is only unique class token, meaning class_token is shared among instances
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # embed class_token to flatten_patches
        x = torch.cat((cls_tokens, patch_x), dim=1)
        # add positional embedding to flatten_patches
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        # loss function(?) since we fix weight of pretrained model and update prompt parameters
        prompt_loss = torch.zeros((1,), requires_grad=True).cuda() # need to fix this line to .to(DEVICE) !!!
        for i,block in enumerate(self.blocks):

            if prompt is not None:
                if train:
                    p_list, loss, x = prompt(q, i, x, train=True, task_id=task_id, possible_task_id=possible_task_id)
                    prompt_loss += loss
                else:
                    p_list, _, x = prompt(q, i, x, train=False, task_id=task_id, possible_task_id=possible_task_id)
            else:
                p_list = None
            # if register_hook_for_block_id == -1 then no need to have register_hook for each Block
            if block_id_for_register_hook == i:
                x = block(x=x, register_hook=True, prompt=p_list)
            else:
                x = block(x=x, register_hook=False, prompt=p_list)

        x = self.norm(x)
        
        return x, prompt_loss

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
        

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
#     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
#         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
#         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
#     if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
#         model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
#         model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

            
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint