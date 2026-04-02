import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from functools import partial
import os
import matplotlib.pyplot as plt


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_patches,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        self.norm1 = norm_layer(dim, elementwise_affine=True)
        self.attn = Attention(
            dim, num_patches,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)

class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], eps=1e-5)


def temporal_variation_all_classes(logits):
    # logits: (T, C)
    diffs = logits[1:] - logits[:-1]              # (T-1, C)
    dist = np.linalg.norm(diffs, axis=1)          # (T-1,)
    return dist.mean(), dist


def top1_top2_margin(logits):
    # logits: (T, C)
    sorted_logits = np.sort(logits, axis=1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    return margin.mean(), margin


def entropy_over_time(logits):
    # logits: (T, C)
    x_shift = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(x_shift)
    probs = probs / probs.sum(axis=1, keepdims=True)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    return entropy.mean(), entropy


def visualize_debug(
    x_before_rnn, x_after_rnn,
    logit_before_raw, logit_after_raw,
    logit_before_norm, logit_after_norm
):
    os.makedirs("visual_debug", exist_ok=True)

    before_feat = x_before_rnn[0].detach().cpu().numpy()       # (L, C_feat)
    after_feat  = x_after_rnn[0].detach().cpu().numpy()        # (L, C_feat)

    logit_b_raw = logit_before_raw[0].detach().cpu().numpy()   # (L, C_cls)
    logit_a_raw = logit_after_raw[0].detach().cpu().numpy()    # (L, C_cls)

    logit_b_norm = logit_before_norm[0].detach().cpu().numpy() # (L, C_cls)
    logit_a_norm = logit_after_norm[0].detach().cpu().numpy()  # (L, C_cls)

    # 1) Feature heatmap
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title("Before RNN (Feature)")
    plt.imshow(before_feat.T, aspect="auto")
    plt.xlabel("Time step")
    plt.ylabel("Feature dim")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("After RNN (Feature)")
    plt.imshow(after_feat.T, aspect="auto")
    plt.xlabel("Time step")
    plt.ylabel("Feature dim")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("visual_debug/feature_before_after_rnn.png", dpi=200)
    plt.close()

    # 2) Heatmap all classes - RAW logits
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.title("RAW logits before RNN (all classes)")
    plt.imshow(logit_b_raw.T, aspect="auto")
    plt.xlabel("Time step")
    plt.ylabel("Class id")
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.title("RAW logits after RNN (all classes)")
    plt.imshow(logit_a_raw.T, aspect="auto")
    plt.xlabel("Time step")
    plt.ylabel("Class id")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("visual_debug/raw_logits_all_classes_heatmap.png", dpi=200)
    plt.close()

    # 3) Heatmap all classes - NORMALIZED logits
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.title("Normalized logits before RNN (all classes)")
    plt.imshow(logit_b_norm.T, aspect="auto")
    plt.xlabel("Time step")
    plt.ylabel("Class id")
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.title("Normalized logits after RNN (all classes)")
    plt.imshow(logit_a_norm.T, aspect="auto")
    plt.xlabel("Time step")
    plt.ylabel("Class id")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("visual_debug/norm_logits_all_classes_heatmap.png", dpi=200)
    plt.close()

    # 4) One representative class - RAW logits
    cls_id = logit_a_raw.mean(axis=0).argmax()
    print(f"[VISUAL] Representative class id = {cls_id}")

    plt.figure(figsize=(14, 4))
    plt.plot(logit_b_raw[:, cls_id], linestyle="solid", marker="o", markersize=4,
             linewidth=2, color="blue", label="Before RNN (RAW)")
    plt.plot(logit_a_raw[:, cls_id], linestyle="solid", marker="s", markersize=4,
             linewidth=2, color="orange", label="After RNN (RAW)")
    plt.title(f"RAW logits over time (class {cls_id})")
    plt.xlabel("Time step")
    plt.ylabel("Logit value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_debug/logit_raw_one_class.png", dpi=200)
    plt.close()

    # 5) One representative class - NORMALIZED logits
    plt.figure(figsize=(14, 4))
    plt.plot(logit_b_norm[:, cls_id], linestyle="solid", marker="o", markersize=4,
             linewidth=2, color="blue", label="Before RNN (Norm)")
    plt.plot(logit_a_norm[:, cls_id], linestyle="solid", marker="s", markersize=4,
             linewidth=2, color="orange", label="After RNN (Norm)")
    plt.title(f"Normalized logits over time (class {cls_id})")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_debug/logit_norm_one_class.png", dpi=200)
    plt.close()

    # 6) Temporal variation
    tv_before_raw,  tv_curve_before_raw  = temporal_variation_all_classes(logit_b_raw)
    tv_after_raw,   tv_curve_after_raw   = temporal_variation_all_classes(logit_a_raw)
    tv_before_norm, tv_curve_before_norm = temporal_variation_all_classes(logit_b_norm)
    tv_after_norm,  tv_curve_after_norm  = temporal_variation_all_classes(logit_a_norm)

    plt.figure(figsize=(14, 4))
    plt.plot(tv_curve_before_raw,  label=f"Before RNN RAW (mean={tv_before_raw:.3f})")
    plt.plot(tv_curve_after_raw,   label=f"After RNN RAW (mean={tv_after_raw:.3f})")
    plt.title("Temporal variation over all classes (RAW logits)")
    plt.xlabel("Time step")
    plt.ylabel("L2 diff between consecutive logits")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_debug/temporal_variation_raw.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(tv_curve_before_norm, label=f"Before RNN Norm (mean={tv_before_norm:.3f})")
    plt.plot(tv_curve_after_norm,  label=f"After RNN Norm (mean={tv_after_norm:.3f})")
    plt.title("Temporal variation over all classes (Normalized logits)")
    plt.xlabel("Time step")
    plt.ylabel("L2 diff between consecutive logits")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_debug/temporal_variation_norm.png", dpi=200)
    plt.close()

    # 7) Top1-Top2 margin
    margin_before_raw,  margin_curve_before_raw  = top1_top2_margin(logit_b_raw)
    margin_after_raw,   margin_curve_after_raw   = top1_top2_margin(logit_a_raw)
    margin_before_norm, margin_curve_before_norm = top1_top2_margin(logit_b_norm)
    margin_after_norm,  margin_curve_after_norm  = top1_top2_margin(logit_a_norm)

    plt.figure(figsize=(14, 4))
    plt.plot(margin_curve_before_raw,  label=f"Before RNN RAW (mean={margin_before_raw:.3f})")
    plt.plot(margin_curve_after_raw,   label=f"After RNN RAW (mean={margin_after_raw:.3f})")
    plt.title("Top-1 vs Top-2 margin over time (RAW logits)")
    plt.xlabel("Time step")
    plt.ylabel("Margin")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_debug/top1_top2_margin_raw.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(margin_curve_before_norm, label=f"Before RNN Norm (mean={margin_before_norm:.3f})")
    plt.plot(margin_curve_after_norm,  label=f"After RNN Norm (mean={margin_after_norm:.3f})")
    plt.title("Top-1 vs Top-2 margin over time (Normalized logits)")
    plt.xlabel("Time step")
    plt.ylabel("Margin")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_debug/top1_top2_margin_norm.png", dpi=200)
    plt.close()

    # 8) Entropy
    ent_before_raw,  ent_curve_before_raw  = entropy_over_time(logit_b_raw)
    ent_after_raw,   ent_curve_after_raw   = entropy_over_time(logit_a_raw)
    ent_before_norm, ent_curve_before_norm = entropy_over_time(logit_b_norm)
    ent_after_norm,  ent_curve_after_norm  = entropy_over_time(logit_a_norm)

    plt.figure(figsize=(14, 4))
    plt.plot(ent_curve_before_raw,  label=f"Before RNN RAW (mean={ent_before_raw:.3f})")
    plt.plot(ent_curve_after_raw,   label=f"After RNN RAW (mean={ent_after_raw:.3f})")
    plt.title("Entropy over time (RAW logits)")
    plt.xlabel("Time step")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_debug/entropy_raw.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(ent_curve_before_norm, label=f"Before RNN Norm (mean={ent_before_norm:.3f})")
    plt.plot(ent_curve_after_norm,  label=f"After RNN Norm (mean={ent_after_norm:.3f})")
    plt.title("Entropy over time (Normalized logits)")
    plt.xlabel("Time step")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("visual_debug/entropy_norm.png", dpi=200)
    plt.close()

    # Print stats
    print("[VISUAL] Saved PNGs to visual_debug/")
    print(f"[STAT][RAW ] Temporal variation: before={tv_before_raw:.6f}, after={tv_after_raw:.6f}")
    print(f"[STAT][NORM] Temporal variation: before={tv_before_norm:.6f}, after={tv_after_norm:.6f}")
    print(f"[STAT][RAW ] Top1-Top2 margin: before={margin_before_raw:.6f}, after={margin_after_raw:.6f}")
    print(f"[STAT][NORM] Top1-Top2 margin: before={margin_before_norm:.6f}, after={margin_after_norm:.6f}")
    print(f"[STAT][RAW ] Entropy: before={ent_before_raw:.6f}, after={ent_after_raw:.6f}")
    print(f"[STAT][NORM] Entropy: before={ent_before_norm:.6f}, after={ent_after_norm:.6f}")


# =========================
# HTR-VT + RNN + CTC
# =========================
class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        nb_cls=80,
        img_size=[512, 32],
        patch_size=[8, 32],
        embed_dim=768,
        depth=4,
        num_heads=6,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        num_layers_RNN=2,
        hidden_dim_RNN=256
    ):
        super().__init__()

        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                      requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, self.num_patches,
                  mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim, elementwise_affine=True)

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim_RNN,
            num_layers=num_layers_RNN,
            batch_first=True,
            bidirectional=True
        )
        self.rnn_proj = nn.Linear(hidden_dim_RNN * 2, embed_dim)
        self.rnn_drop = nn.Dropout(0.2)

        self.head = nn.Linear(embed_dim, nb_cls)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:, idx:idx + max_span_length, :] = 0
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        x_masked = x * mask + (1 - mask) * self.mask_token
        return x_masked

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        # Patch embedding
        x = self.layer_norm(x)
        x = self.patch_embed(x)
        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # (B, L, C)

        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)

        x = x + self.pos_embed

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Before RNN
        x_before_rnn = x.detach()
        logit_before_raw  = self.head(x_before_rnn)
        logit_before_norm = self.layer_norm(logit_before_raw)

        # RNN
        x, _ = self.rnn(x)
        x = self.rnn_proj(x)


        x_after_rnn = x.detach()
        logit_after_raw  = self.head(x_after_rnn)
        logit_after_norm = self.layer_norm(logit_after_raw)

        # CTC logits
        logits = logit_after_raw

        # Visualization (only once)
        if not hasattr(self, "_visualized"):
            self._visualized = True
            visualize_debug(
                x_before_rnn, x_after_rnn,
                logit_before_raw, logit_after_raw,
                logit_before_norm, logit_after_norm
            )

        return logits


def create_model(nb_cls, img_size, num_layer_RNN, hidden_dim_RNN, **kwargs):
    return MaskedAutoencoderViT(
        nb_cls=nb_cls,
        img_size=img_size,
        patch_size=(4, 64),
        embed_dim=768,
        depth=4,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_layers_RNN=num_layer_RNN,
        hidden_dim_RNN=hidden_dim_RNN,
        **kwargs
    )