import os
import sys
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

# Allow importing from project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sam3.model.vitdet import ViT


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tokens_to_map(t: torch.Tensor, batch_size: int, embed_dim: int = 1024) -> torch.Tensor:
    """Convert transformer outputs to [B,C,H,W]."""
    if t.dim() == 4:
        if t.shape[1] == embed_dim:
            return t
        if t.shape[3] == embed_dim:
            return t.permute(0, 3, 1, 2).contiguous()

        # Fallback: infer whether it's [B,C,H,W] or [B,H,W,C]
        if t.shape[1] > t.shape[2] and t.shape[1] > t.shape[3]:
            return t
        return t.permute(0, 3, 1, 2).contiguous()

    if t.dim() == 3:
        tokens = t.shape[1]
        side = int(tokens ** 0.5)
        if side * side != tokens:
            raise ValueError(f"Token length {tokens} is not a perfect square.")
        return t.permute(0, 2, 1).contiguous().view(batch_size, -1, side, side)

    raise ValueError(f"Unsupported tensor shape: {tuple(t.shape)}")


def _infer_dino_layout(dino_model: nn.Module, low_res_size: int) -> Tuple[int, int, int, int]:
    """Return (expected_tokens, prefix_tokens, patch_size, grid_size)."""
    patch = 16
    if hasattr(dino_model, "patch_embed") and hasattr(dino_model.patch_embed, "patch_size"):
        ps = dino_model.patch_embed.patch_size
        patch = int(ps[0] if isinstance(ps, (tuple, list)) else ps)

    grid = low_res_size // patch
    expected_tokens = grid * grid

    num_register = 0
    for attr in ("num_register_tokens", "n_register_tokens", "register_tokens"):
        if hasattr(dino_model, attr):
            value = getattr(dino_model, attr)
            if isinstance(value, int):
                num_register = value
                break
            if torch.is_tensor(value):
                num_register = int(value.shape[1])
                break

    has_cls = 0
    if hasattr(dino_model, "cls_token") and torch.is_tensor(getattr(dino_model, "cls_token")):
        has_cls = 1
    for flag in ("use_cls_token", "use_clstoken", "with_cls_token"):
        if hasattr(dino_model, flag) and bool(getattr(dino_model, flag)):
            has_cls = 1

    return expected_tokens, has_cls + num_register, patch, grid


def _strip_patch_tokens(tokens: torch.Tensor, expected_tokens: int, prefix_tokens: int) -> torch.Tensor:
    """Ensure token tensor is [B, expected_tokens, C] by dropping cls/register/prefix tokens if present."""
    if tokens.dim() != 3:
        raise ValueError(f"Expected [B,N,C], got {tuple(tokens.shape)}")

    total = tokens.shape[1]
    if total == expected_tokens + prefix_tokens:
        return tokens[:, prefix_tokens:, :]
    if total == expected_tokens:
        return tokens
    if total > expected_tokens:
        return tokens[:, total - expected_tokens :, :]

    raise ValueError(f"DINO tokens too few: total={total}, expected={expected_tokens}")


def _to_patch_token_sequence(tokens: torch.Tensor) -> torch.Tensor:
    """Convert DINO patch outputs to [B, N, C]."""
    if tokens.dim() == 3:
        return tokens
    if tokens.dim() != 4:
        raise ValueError(f"Unsupported patch token shape: {tuple(tokens.shape)}")

    # [B, C, H, W]
    if tokens.shape[1] >= tokens.shape[2] and tokens.shape[1] >= tokens.shape[3]:
        return tokens.flatten(2).transpose(1, 2).contiguous()
    # [B, H, W, C]
    if tokens.shape[3] >= tokens.shape[1] and tokens.shape[3] >= tokens.shape[2]:
        return tokens.reshape(tokens.shape[0], -1, tokens.shape[3]).contiguous()

    raise ValueError(f"Cannot infer token layout for shape: {tuple(tokens.shape)}")


def _ordinal_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """CORAL logits [B,K-1] -> probabilities [B,K]."""
    p_gt = torch.sigmoid(logits)
    bsz, k_minus_1 = p_gt.shape
    k = k_minus_1 + 1
    probs = logits.new_zeros((bsz, k))
    probs[:, 0] = 1.0 - p_gt[:, 0]
    for cls in range(1, k - 1):
        probs[:, cls] = p_gt[:, cls - 1] - p_gt[:, cls]
    probs[:, k - 1] = p_gt[:, -1]
    probs = probs.clamp_min(1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs


class LoRAQKV(nn.Module):
    """LoRA adapter for a fused qkv projection. Updates Q and V only."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRAQKV expects nn.Linear, got {type(base)}")
        if base.out_features % 3 != 0:
            raise ValueError(f"qkv out_features must be divisible by 3, got {base.out_features}")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout))

        dim = base.out_features // 3
        self.lora_q_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_q_B = nn.Linear(self.rank, dim, bias=False)
        self.lora_v_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_v_B = nn.Linear(self.rank, dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.zeros_(self.lora_v_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Non-inplace version (safer for autograd/compile)
        qkv = self.base(x)
        if self.rank <= 0:
            return qkv

        x_drop = self.dropout(x)
        q_update = self.lora_q_B(self.lora_q_A(x_drop)) * self.scale
        v_update = self.lora_v_B(self.lora_v_A(x_drop)) * self.scale

        q, k, v = qkv.chunk(3, dim=-1)
        q = q + q_update
        v = v + v_update
        return torch.cat([q, k, v], dim=-1)


def _inject_dino_lora(model: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    replaced = 0
    for _, child in model.named_children():
        if hasattr(child, "qkv") and isinstance(getattr(child, "qkv"), nn.Linear):
            child.qkv = LoRAQKV(child.qkv, rank=rank, alpha=alpha, dropout=dropout)
            replaced += 1
        replaced += _inject_dino_lora(child, rank, alpha, dropout)
    return replaced


def _create_sam3_vit(img_size: int) -> ViT:
    return ViT(
        img_size=img_size,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=None,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        gn_groups = 8 if out_channels % 8 == 0 else 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConditionedAdapter(nn.Module):
    """Light adapter for each SAM block, conditioned by DINO global vector + optional spatial cond map."""

    def __init__(self, block: nn.Module, dino_dim: int, bottleneck: int = 32):
        super().__init__()
        self.block = block

        dim = None
        if hasattr(block, "attn") and hasattr(block.attn, "qkv") and hasattr(block.attn.qkv, "in_features"):
            dim = int(block.attn.qkv.in_features)
        elif hasattr(block, "norm1") and hasattr(block.norm1, "normalized_shape"):
            dim = int(block.norm1.normalized_shape[0])
        if dim is None:
            raise AttributeError("Cannot infer SAM block embedding dim.")
        self.dim = dim

        self.prompt = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
            nn.GELU(),
        )
        self.cond = nn.Sequential(
            nn.Linear(dino_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )
        # Spatial FiLM: produce gamma/beta for each spatial location
        self.cond_spatial = nn.Sequential(
            nn.Linear(dino_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim * 2),
        )

        self._cond_cache: Optional[torch.Tensor] = None          # [B, dino_dim]
        self._cond_map_cache: Optional[torch.Tensor] = None      # [B, Hc, Wc, dino_dim] or [B, Nc, dino_dim]
        self._init_weights()

    def _init_weights(self):
        def _init(module: nn.Module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.prompt.apply(_init)
        self.cond.apply(_init)
        self.cond_spatial.apply(_init)

    def set_condition(self, cond: torch.Tensor):
        self._cond_cache = cond
        self._cond_map_cache = None

    def set_condition_map(self, cond: torch.Tensor, cond_map: Optional[torch.Tensor] = None):
        self._cond_cache = cond
        self._cond_map_cache = cond_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cond_cache is None:
            raise RuntimeError("ConditionedAdapter requires set_condition(cond) before forward.")

        # Heuristic: if input is [B, C, H, W] (BCHW), convert to BHWC.
        is_bchw = False
        if x.dim() == 4 and x.shape[1] == self.dim:
            x = x.permute(0, 2, 3, 1).contiguous()
            is_bchw = True

        p = self.prompt(x)
        c_vec = self.cond(self._cond_cache)

        if x.dim() == 3:
            c = c_vec.unsqueeze(1)
        elif x.dim() == 4:
            c = c_vec.view(x.shape[0], 1, 1, self.dim)
        else:
            raise RuntimeError(f"Unsupported adapter input shape: {tuple(x.shape)}")

        result = x + p + c

        # Optional spatial FiLM
        if self._cond_map_cache is not None:
            cond_map = self._cond_map_cache

            # Normalize cond_map to BHWC: [B, Hc, Wc, dino_dim]
            if cond_map.dim() == 3:
                side = int(cond_map.shape[1] ** 0.5)
                if side * side != cond_map.shape[1]:
                    raise RuntimeError("ConditionedAdapter cond_map tokens not square.")
                cond_map = cond_map.view(cond_map.shape[0], side, side, cond_map.shape[2])
            elif cond_map.dim() == 4:
                # Accept BHWC or BCHW for cond_map
                if cond_map.shape[-1] == self.cond_spatial[0].in_features:
                    pass
                elif cond_map.shape[1] == self.cond_spatial[0].in_features:
                    cond_map = cond_map.permute(0, 2, 3, 1).contiguous()
                else:
                    raise RuntimeError(f"Unsupported cond_map shape: {tuple(cond_map.shape)}")
            else:
                raise RuntimeError(f"Unsupported cond_map shape: {tuple(cond_map.shape)}")

            if result.dim() == 3:
                # result: [B, N, dim] -> [B, H, W, dim]
                B, N, D = result.shape
                side = int(N ** 0.5)
                if side * side != N:
                    raise RuntimeError("ConditionedAdapter token input not square for spatial conditioning.")
                result_hw = result.view(B, side, side, D)

                cond_hw = cond_map
                if cond_hw.shape[1] != side or cond_hw.shape[2] != side:
                    cond_hw = F.interpolate(
                        cond_hw.permute(0, 3, 1, 2),
                        size=(side, side),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

                spatial_film = self.cond_spatial(cond_hw)
                gamma, beta = spatial_film.chunk(2, dim=-1)

                # Stabilize FiLM strength
                gamma = torch.tanh(gamma)
                result_hw = result_hw * (1.0 + 0.5 * gamma) + beta

                result = result_hw.view(B, N, D)

            else:
                # result: [B, H, W, dim]
                B, H, W, D = result.shape
                cond_hw = cond_map
                if cond_hw.shape[1] != H or cond_hw.shape[2] != W:
                    cond_hw = F.interpolate(
                        cond_hw.permute(0, 3, 1, 2),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

                spatial_film = self.cond_spatial(cond_hw)
                gamma, beta = spatial_film.chunk(2, dim=-1)
                gamma = torch.tanh(gamma)
                result = result * (1.0 + 0.5 * gamma) + beta

        if is_bchw:
            result = result.permute(0, 3, 1, 2).contiguous()

        return self.block(result)


# ==================== Disease Region Clustering ====================

class DiseaseRegionClustering(nn.Module):
    """Learn K trainable prototypes and softly assign DINO patch tokens to them."""

    def __init__(
        self,
        dino_dim: int = 1024,
        num_prototypes: int = 6,
        hidden_dim: int = 256,
        temperature: float = 0.1,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)

        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, dino_dim))
        nn.init.orthogonal_(self.prototypes)

        self.feat_proj = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.proto_proj = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,
        grid_size: int,
        roi_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if patch_tokens.dim() != 3:
            raise ValueError(f"patch_tokens must be [B,N,C], got {tuple(patch_tokens.shape)}")
        B, N, C = patch_tokens.shape
        if grid_size * grid_size != N:
            side = int(N ** 0.5)
            if side * side != N:
                raise ValueError(f"Token length {N} is not a perfect square, cannot form cluster map.")
            grid_size = side

        feat = self.feat_proj(patch_tokens)                 # [B,N,hidden]
        proto_feat = self.proto_proj(self.prototypes)       # [K,hidden]

        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(proto_feat, dim=-1)
        similarity = torch.matmul(feat_norm, proto_norm.t()) / self.temperature  # [B,N,K]

        cluster_probs = F.softmax(similarity, dim=-1)       # [B,N,K]
        cluster_assignment = cluster_probs.argmax(dim=-1)   # [B,N]

        token_weight = None
        if roi_weight is not None:
            if roi_weight.dim() == 4:
                token_weight = roi_weight.squeeze(1).reshape(B, -1)
            elif roi_weight.dim() == 3:
                token_weight = roi_weight.reshape(B, -1)
            elif roi_weight.dim() == 2:
                token_weight = roi_weight
            else:
                raise ValueError(f"Unsupported roi_weight shape: {tuple(roi_weight.shape)}")
            if token_weight.shape[1] != N:
                raise ValueError(f"roi_weight tokens={token_weight.shape[1]} but patch tokens={N}")
            token_weight = token_weight.clamp(0.0, 1.0)

        weighted_probs = cluster_probs if token_weight is None else cluster_probs * token_weight.unsqueeze(-1)

        cluster_map = cluster_probs.permute(0, 2, 1).contiguous().view(B, self.num_prototypes, grid_size, grid_size)
        cluster_map_weighted = weighted_probs.permute(0, 2, 1).contiguous().view(B, self.num_prototypes, grid_size, grid_size)

        cluster_weights = weighted_probs.permute(0, 2, 1)  # [B,K,N]
        cluster_mass = cluster_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B,K,1]
        cluster_feat = torch.matmul(cluster_weights, patch_tokens) / cluster_mass  # [B,K,C]

        return {
            "cluster_map": cluster_map,
            "cluster_map_weighted": cluster_map_weighted,
            "cluster_feat": cluster_feat,
            "cluster_assignment": cluster_assignment,
            "proto_similarity": similarity,
            "cluster_probs": cluster_probs,
            "cluster_mass": cluster_mass.squeeze(-1),
            "cluster_token_weight": token_weight,
        }

    def prototype_diversity_loss(self) -> torch.Tensor:
        """Orthogonality regularizer to prevent prototype collapse."""
        if self.num_prototypes <= 1:
            return self.prototypes.new_tensor(0.0)
        proto = F.normalize(self.prototypes, dim=-1)
        gram = proto @ proto.t()
        ident = torch.eye(self.num_prototypes, device=gram.device, dtype=gram.dtype)
        diff = gram - ident
        return (diff * diff).sum() / (self.num_prototypes * (self.num_prototypes - 1))


class RegionFusionModule(nn.Module):
    """Fuse prototype assignment maps into SAM features."""

    def __init__(self, sam_dim: int = 128, cluster_dim: int = 6, fusion_dim: int = 128):
        super().__init__()
        self.cluster_upsample = nn.Sequential(
            nn.Conv2d(cluster_dim, fusion_dim // 2, 1),
            nn.GroupNorm(4, fusion_dim // 2),
            nn.GELU(),
        )
        gn_groups = 8 if fusion_dim % 8 == 0 else 4
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sam_dim + fusion_dim // 2, fusion_dim, 3, padding=1),
            nn.GroupNorm(gn_groups, fusion_dim),
            nn.GELU(),
            nn.Conv2d(fusion_dim, sam_dim, 1),
        )

    def forward(self, sam_feat: torch.Tensor, cluster_map: torch.Tensor, roi_gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        H, W = sam_feat.shape[2:]
        cluster_upsampled = F.interpolate(cluster_map, size=(H, W), mode="bilinear", align_corners=False)

        if roi_gate is not None:
            gate = F.interpolate(roi_gate, size=(H, W), mode="bilinear", align_corners=False)
            cluster_upsampled = cluster_upsampled * gate

        cluster_feat = self.cluster_upsample(cluster_upsampled)
        fused = torch.cat([sam_feat, cluster_feat], dim=1)
        output = self.fusion_conv(fused)
        return sam_feat + output


class PrototypeAttentionPooling(nn.Module):
    """Pool K prototype features into fixed-dim vector for grade head."""

    def __init__(self, feat_dim: int, out_dim: int):
        super().__init__()
        hidden = max(out_dim, feat_dim // 4)
        self.score = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, cluster_feat: torch.Tensor, cluster_mass: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(cluster_feat).squeeze(-1)  # [B,K]
        if cluster_mass is not None:
            logits = logits + torch.log(cluster_mass.clamp_min(1e-6))
        attn = F.softmax(logits, dim=1)
        pooled = (attn.unsqueeze(-1) * cluster_feat).sum(dim=1)
        return self.proj(pooled), attn


class BidirectionalCrossTaskAttention(nn.Module):
    """Two-way cross-task interaction between segmentation feature map and grade token."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dim = int(dim)
        self.seg_norm = nn.LayerNorm(dim)
        self.grade_norm = nn.LayerNorm(dim)

        self.cls_to_seg = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.seg_to_cls = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

        self.seg_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.grade_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        # Cache fp32 positional embeddings
        self._pos_cache: Dict[Tuple[int, int, int, torch.device], torch.Tensor] = {}

    @staticmethod
    def _build_1d_sincos_pos(dim: int, positions: torch.Tensor) -> torch.Tensor:
        if dim <= 0:
            return positions.new_zeros((positions.shape[0], 0))
        half = dim // 2
        if half == 0:
            return positions.new_zeros((positions.shape[0], dim))
        omega = torch.arange(half, device=positions.device, dtype=positions.dtype)
        omega = 1.0 / (10000 ** (omega / float(half)))
        out = positions[:, None] * omega[None, :]
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _get_2d_pos(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, self.dim, device)
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached

        dtype = torch.float32
        dim_h = self.dim // 2
        dim_w = self.dim - dim_h

        grid_h = torch.arange(height, device=device, dtype=dtype)
        grid_w = torch.arange(width, device=device, dtype=dtype)

        emb_h = self._build_1d_sincos_pos(dim_h, grid_h)
        emb_w = self._build_1d_sincos_pos(dim_w, grid_w)

        pos = torch.cat(
            [
                emb_h[:, None, :].expand(height, width, -1),
                emb_w[None, :, :].expand(height, width, -1),
            ],
            dim=2,
        ).view(1, height * width, self.dim)

        self._pos_cache[key] = pos
        return pos

    def forward(self, seg_feat: torch.Tensor, grade_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, height, width = seg_feat.shape
        seg_tokens = seg_feat.flatten(2).transpose(1, 2).contiguous()  # [B,N,C]

        pos = self._get_2d_pos(height, width, seg_tokens.device).to(seg_tokens.dtype)
        seg_tokens = seg_tokens + pos

        grade_seq = grade_token.unsqueeze(1)  # [B,1,C]

        seg_q = self.seg_norm(seg_tokens)
        grade_kv = self.grade_norm(grade_seq)
        cls_msg, _ = self.cls_to_seg(seg_q, grade_kv, grade_kv, need_weights=False)
        seg_tokens = seg_tokens + cls_msg
        seg_tokens = seg_tokens + self.seg_ffn(seg_tokens)

        grade_q = self.grade_norm(grade_seq)
        seg_kv = self.seg_norm(seg_tokens)
        seg_msg, _ = self.seg_to_cls(grade_q, seg_kv, seg_kv, need_weights=False)
        grade_seq = grade_seq + seg_msg
        grade_seq = grade_seq + self.grade_ffn(grade_seq)

        grade_token = grade_seq.squeeze(1)
        gate = self.channel_gate(grade_token).view(bsz, channels, 1, 1)

        seg_feat_out = seg_tokens.transpose(1, 2).reshape(bsz, channels, height, width).contiguous()
        seg_feat_out = seg_feat_out * (1.0 + gate)
        return seg_feat_out, grade_token


class SegGradeConsistencyCorrector(nn.Module):
    """Adaptive correction when segmentation-derived and head-derived grades are inconsistent."""

    def __init__(
        self,
        grade_values_norm: torch.Tensor,
        grade_bounds: Sequence[Tuple[float, float]],
        hidden_dim: int = 32,
        temperature: float = 0.12,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = float(temperature)

        centers = [0.5 * (lo + hi) for lo, hi in grade_bounds]
        widths = [max(hi - lo, 1e-3) for lo, hi in grade_bounds]

        self.register_buffer("grade_values_norm", grade_values_norm.clone(), persistent=False)
        self.register_buffer("grade_centers", torch.tensor(centers, dtype=torch.float32), persistent=False)
        self.register_buffer("grade_widths", torch.tensor(widths, dtype=torch.float32), persistent=False)

        self.fuse_weight_head = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def ratio_to_soft_probs(self, ratio: torch.Tensor) -> torch.Tensor:
        z = (ratio.unsqueeze(1) - self.grade_centers.unsqueeze(0)).abs()
        z = z / (self.grade_widths.unsqueeze(0) + 1e-6)
        logits = -z / self.temperature
        return F.softmax(logits, dim=1)

    def forward(
        self,
        grade_probs_head: torch.Tensor,
        grade_score_head: torch.Tensor,
        disease_ratio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ratio_probs = self.ratio_to_soft_probs(disease_ratio)
        ratio_score = (ratio_probs * self.grade_values_norm.unsqueeze(0)).sum(dim=1)
        inconsistency = (grade_score_head - ratio_score).abs()

        entropy_head = -(grade_probs_head.clamp_min(1e-8) * grade_probs_head.clamp_min(1e-8).log()).sum(dim=1)
        entropy_ratio = -(ratio_probs.clamp_min(1e-8) * ratio_probs.clamp_min(1e-8).log()).sum(dim=1)

        fuse_inputs = torch.stack(
            [grade_score_head, ratio_score, inconsistency, disease_ratio, entropy_head, entropy_ratio],
            dim=1,
        )
        weight_head = torch.sigmoid(self.fuse_weight_head(fuse_inputs)).squeeze(1)

        corrected_probs = weight_head.unsqueeze(1) * grade_probs_head + (1.0 - weight_head).unsqueeze(1) * ratio_probs
        corrected_probs = corrected_probs / corrected_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        corrected_score = (corrected_probs * self.grade_values_norm.unsqueeze(0)).sum(dim=1)

        return {
            "ratio_grade_probs": ratio_probs,
            "ratio_grade_score": ratio_score,
            "inconsistency_score": inconsistency,
            "consistency_weight_head": weight_head,
            "corrected_probs": corrected_probs,
            "corrected_score": corrected_score,
        }


class BoundaryPropagationUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.prev_feat_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.prev_boundary_proj = nn.Conv2d(1, channels, 1, bias=False)
        self.fuse = ConvBlock(channels * 3, channels)
        self.boundary_head = nn.Conv2d(channels, 1, 3, padding=1)
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(1, channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat: torch.Tensor,
        prev_feat: Optional[torch.Tensor],
        prev_boundary: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, channels, height, width = feat.shape

        if prev_feat is None:
            prev_feat_proj = feat.new_zeros((bsz, channels, height, width))
        else:
            prev_feat_up = F.interpolate(prev_feat, size=(height, width), mode="bilinear", align_corners=False)
            prev_feat_proj = self.prev_feat_proj(prev_feat_up)

        if prev_boundary is None:
            prev_boundary_proj = feat.new_zeros((bsz, channels, height, width))
        else:
            prev_boundary_up = F.interpolate(prev_boundary, size=(height, width), mode="bilinear", align_corners=False)
            prev_boundary_proj = self.prev_boundary_proj(prev_boundary_up)

        fused = self.fuse(torch.cat([feat, prev_feat_proj, prev_boundary_proj], dim=1))
        boundary_logit = self.boundary_head(fused)
        refined = feat + fused * self.boundary_gate(boundary_logit)
        return refined, boundary_logit


class HierarchicalBoundaryPropagation(nn.Module):
    """Coarse-to-fine boundary refinement with cross-scale fusion."""

    def __init__(self, channels: int, num_levels: int = 4):
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be >= 2")
        self.num_levels = int(num_levels)
        self.units = nn.ModuleList([BoundaryPropagationUnit(channels) for _ in range(num_levels)])
        gn_groups = 8 if channels % 8 == 0 else 4
        self.scale_fuse = nn.Sequential(
            nn.Conv2d(channels * num_levels, channels, 1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.GELU(),
        )
        self.boundary_scale_weights = nn.Parameter(torch.zeros(num_levels))

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        if len(feats) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} feature levels, got {len(feats)}")

        refined: List[Optional[torch.Tensor]] = [None] * self.num_levels
        boundary_logits: List[Optional[torch.Tensor]] = [None] * self.num_levels
        prev_feat = None
        prev_boundary = None

        for idx in range(self.num_levels - 1, -1, -1):
            feat, boundary = self.units[idx](feats[idx], prev_feat, prev_boundary)
            refined[idx] = feat
            boundary_logits[idx] = boundary
            prev_feat = feat
            prev_boundary = boundary

        refined_feats = [x for x in refined if x is not None]
        logits = [x for x in boundary_logits if x is not None]
        if len(refined_feats) != self.num_levels or len(logits) != self.num_levels:
            raise RuntimeError("Boundary propagation produced incomplete outputs.")

        target_size = refined_feats[0].shape[2:]
        upsampled_feats = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in refined_feats]
        boundary_context = self.scale_fuse(torch.cat(upsampled_feats, dim=1))
        refined_feats[0] = refined_feats[0] + boundary_context

        w = F.softmax(self.boundary_scale_weights, dim=0)
        fused_boundary = 0.0
        for i, logit in enumerate(logits):
            up = F.interpolate(logit, size=target_size, mode="bilinear", align_corners=False)
            fused_boundary = fused_boundary + w[i] * up

        return refined_feats, logits, fused_boundary, boundary_context


# ==================== Main Model ====================

class DMNetWithClustering(nn.Module):
    """Enhanced soybean downy mildew grading model with clustering + cross-task attention."""

    def __init__(
        self,
        sam3_checkpoint_path: Optional[str] = None,
        dinov3_weight_path: Optional[str] = None,
        dinov3_local_path: str = "./dinov3",
        dinov3_model_name: str = "dinov3_vitl16",
        img_size: int = 512,
        low_res_size: int = 448,
        num_classes: int = 3,
        sam_stage_ids: Sequence[int] = (7, 15, 23, 31),
        fuse_dim: int = 128,
        adapter_bottleneck: int = 32,
        grade_values: Sequence[int] = (0, 1, 3, 5, 7, 9),
        leaf_class_id: int = 1,
        disease_class_ids: Sequence[int] = (2,),
        num_prototypes: int = 6,
        use_clustering: bool = True,
        cluster_temperature: float = 0.1,
        use_cross_task_attention: bool = True,
        use_consistency_correction: bool = True,
        use_boundary_propagation: bool = True,
        use_soft_roi_cluster: bool = True,
        use_prototype_attention_pooling: bool = True,
        cross_attention_heads: int = 4,
        consistency_temperature: float = 0.12,
        roi_ring_kernel: int = 7,
        roi_center_weight: float = 0.8,
        roi_ring_weight: float = 0.0,
        roi_use_detach: bool = True,
        roi_use_confidence: bool = True,
        roi_use_anomaly: bool = True,
        roi_anomaly_power: float = 1.0,
        grade_use_disease_weighted_pool: bool = True,
        grade_pool_detach: bool = True,
        use_dino_cond_refine: bool = True,
        use_leaf_proxy_head: bool = True,
        leaf_proxy_hidden: int = 256,
        use_roi_gate_in_fusion: bool = False,
        dino_intermediate_layer: int = 11,
        compare_last_layer_head: bool = True,
        cluster_token_source: str = "mid",
        cluster_roi_source: str = "anomaly_then_leaf",
        detach_shared_for_last_head: bool = True,
        dino_lora_rank: int = 0,
        dino_lora_alpha: float = 8.0,
        dino_lora_dropout: float = 0.0,
        min_backbone_load_ratio: float = 0.0,
    ):
        super().__init__()
        if len(sam_stage_ids) != 4:
            raise ValueError("sam_stage_ids must contain 4 stages.")

        self.high_res_size = int(img_size)
        self.low_res_size = int(low_res_size)

        self.num_classes = int(num_classes)
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2.")
        self.fuse_dim = int(fuse_dim)

        self.sam_stage_ids = [int(x) for x in sam_stage_ids]
        self.use_clustering = bool(use_clustering)
        self.use_cross_task_attention = bool(use_cross_task_attention)
        self.use_consistency_correction = bool(use_consistency_correction)
        self.use_boundary_propagation = bool(use_boundary_propagation)
        self.use_soft_roi_cluster = bool(use_soft_roi_cluster)
        self.use_prototype_attention_pooling = bool(use_prototype_attention_pooling)

        self.roi_ring_kernel = max(3, int(roi_ring_kernel))
        if self.roi_ring_kernel % 2 == 0:
            self.roi_ring_kernel += 1
        self.roi_center_weight = float(roi_center_weight)
        self.roi_ring_weight = float(roi_ring_weight)

        self.roi_use_detach = bool(roi_use_detach)
        self.roi_use_confidence = bool(roi_use_confidence)
        self.roi_use_anomaly = bool(roi_use_anomaly)
        self.roi_anomaly_power = float(roi_anomaly_power)

        self.grade_use_disease_weighted_pool = bool(grade_use_disease_weighted_pool)
        self.grade_pool_detach = bool(grade_pool_detach)

        self.use_dino_cond_refine = bool(use_dino_cond_refine)
        self.use_leaf_proxy_head = bool(use_leaf_proxy_head)
        self.use_roi_gate_in_fusion = bool(use_roi_gate_in_fusion)

        self.dino_intermediate_layer = int(dino_intermediate_layer)
        self.compare_last_layer_head = bool(compare_last_layer_head)

        self.cluster_token_source = str(cluster_token_source).strip().lower()
        if self.cluster_token_source not in {"mid", "last", "blend"}:
            raise ValueError(f"cluster_token_source must be one of mid/last/blend, got {cluster_token_source}")

        self.cluster_roi_source = str(cluster_roi_source).strip().lower()
        if self.cluster_roi_source not in {"none", "anomaly", "leaf", "anomaly_then_leaf"}:
            raise ValueError(
                "cluster_roi_source must be one of none/anomaly/leaf/anomaly_then_leaf, "
                f"got {cluster_roi_source}"
            )

        self.detach_shared_for_last_head = bool(detach_shared_for_last_head)

        self.dino_lora_rank = int(dino_lora_rank)
        self.dino_lora_alpha = float(dino_lora_alpha)
        self.dino_lora_dropout = float(dino_lora_dropout)
        self.use_dino_lora = self.dino_lora_rank > 0
        self.min_backbone_load_ratio = float(min_backbone_load_ratio)

        self.grade_values = tuple(int(v) for v in grade_values)
        if len(self.grade_values) < 2:
            raise ValueError("grade_values must contain at least two values.")
        self.num_grade_levels = len(self.grade_values)

        self.leaf_class_id = int(leaf_class_id)
        self.disease_class_ids = tuple(int(v) for v in disease_class_ids)
        if not self.disease_class_ids:
            raise ValueError("disease_class_ids cannot be empty.")

        self.grade_delta = 0.001
        self.grade_bounds = [
            (0.0, self.grade_delta),
            (self.grade_delta, 0.05),
            (0.05, 0.10),
            (0.10, 0.20),
            (0.20, 0.50),
            (0.50, 1.0),
        ]
        if len(self.grade_bounds) != self.num_grade_levels:
            raise ValueError("grade_bounds length must match grade_values length.")

        if not (0 <= self.leaf_class_id < self.num_classes):
            raise ValueError(f"leaf_class_id must be in [0, {self.num_classes - 1}].")
        invalid_disease = [cid for cid in self.disease_class_ids if cid < 0 or cid >= self.num_classes]
        if invalid_disease:
            raise ValueError(f"disease_class_ids contains out-of-range ids: {invalid_disease}")

        norm_grade_values = torch.tensor(self.grade_values, dtype=torch.float32) / float(self.grade_values[-1])
        self.register_buffer("grade_values_norm", norm_grade_values, persistent=True)

        # ---------- DINOv3 ----------
        self.dino = torch.hub.load(
            repo_or_dir=dinov3_local_path,
            model=dinov3_model_name,
            source="local",
            pretrained=False,
            trust_repo=True,
        )

        if dinov3_weight_path and os.path.exists(dinov3_weight_path):
            ckpt = torch.load(dinov3_weight_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                ckpt = ckpt["state_dict"]

            model_sd = self.dino.state_dict()

            # Try direct match
            loadable = {k: v for k, v in ckpt.items() if k in model_sd and model_sd[k].shape == v.shape}
            if len(loadable) == 0:
                # Try stripping "module."
                ckpt2 = {k.replace("module.", "", 1): v for k, v in ckpt.items() if isinstance(k, str) and k.startswith("module.")}
                loadable = {k: v for k, v in ckpt2.items() if k in model_sd and model_sd[k].shape == v.shape}

            model_sd.update(loadable)
            self.dino.load_state_dict(model_sd, strict=False)
            load_ratio = len(loadable) / max(1, len(model_sd))
            print(f"[DINO] loaded {len(loadable)} tensors from {dinov3_weight_path} (ratio={load_ratio:.2%})")
            if self.min_backbone_load_ratio > 0 and load_ratio < self.min_backbone_load_ratio:
                print(f"[DINO] warning: load ratio below threshold ({self.min_backbone_load_ratio:.2%}).")
        else:
            print("[DINO] warning: weight path not found, using current init")

        for p in self.dino.parameters():
            p.requires_grad = False

        self.dino_num_blocks = len(self.dino.blocks) if hasattr(self.dino, "blocks") else 0

        if self.use_dino_lora:
            replaced = _inject_dino_lora(
                self.dino,
                rank=self.dino_lora_rank,
                alpha=self.dino_lora_alpha,
                dropout=self.dino_lora_dropout,
            )
            if replaced == 0:
                print("[DINO] warning: no attention qkv layers found for LoRA injection.")
            else:
                print(f"[DINO] LoRA enabled: rank={self.dino_lora_rank}, alpha={self.dino_lora_alpha}, dropout={self.dino_lora_dropout}, layers={replaced}")

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.low_res_size, self.low_res_size)
            feats = self._dino_forward_features(dummy)
            patch_tokens = feats.get("x_norm_patchtokens") if isinstance(feats, dict) else feats
            if patch_tokens is None:
                raise KeyError("DINO forward_features must return 'x_norm_patchtokens'.")
            self.dino_dim = int(patch_tokens.shape[-1])

        if self.dino_num_blocks > 0:
            self.dino_intermediate_layer = self._resolve_dino_layer_index(self.dino_intermediate_layer)
            if self.dino_intermediate_layer == self.dino_num_blocks - 1:
                print("[DINO] warning: dino_intermediate_layer points to the last block; middle-vs-last comparison is weakened.")
        else:
            self.dino_intermediate_layer = -1
            print("[DINO] warning: cannot infer transformer depth; fallback to final tokens only.")

        # ---------- Clustering ----------
        if self.use_clustering:
            self.clustering = DiseaseRegionClustering(
                dino_dim=self.dino_dim,
                num_prototypes=int(num_prototypes),
                temperature=float(cluster_temperature),
            )
            self.num_prototypes = int(num_prototypes)
        else:
            self.clustering = None
            self.num_prototypes = 0

        # ---------- SAM3 ----------
        self.sam = _create_sam3_vit(self.high_res_size)
        self.sam_patch_size = None
        if hasattr(self.sam, "patch_embed") and hasattr(self.sam.patch_embed, "proj"):
            stride = self.sam.patch_embed.proj.stride
            self.sam_patch_size = int(stride[0] if isinstance(stride, (tuple, list)) else stride)
        if self.sam_patch_size is None or self.sam_patch_size <= 0:
            self.sam_patch_size = 14

        if sam3_checkpoint_path and os.path.exists(sam3_checkpoint_path):
            ckpt = torch.load(sam3_checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                ckpt = ckpt["state_dict"]

            extracted: Dict[str, torch.Tensor] = {}
            for k, v in ckpt.items():
                if isinstance(k, str) and "detector.backbone.vision_backbone.trunk" in k and "freqs_cis" not in k:
                    nk = k[len("detector.backbone.vision_backbone.trunk.") :]
                    extracted[nk] = v

            model_sd = self.sam.state_dict()
            loadable = {k: v for k, v in extracted.items() if k in model_sd and model_sd[k].shape == v.shape}
            if len(loadable) == 0:
                extracted2 = {k.replace("module.", "", 1): v for k, v in extracted.items() if k.startswith("module.")}
                loadable = {k: v for k, v in extracted2.items() if k in model_sd and model_sd[k].shape == v.shape}

            model_sd.update(loadable)
            self.sam.load_state_dict(model_sd, strict=False)
            load_ratio = len(loadable) / max(1, len(model_sd))
            print(f"[SAM3] loaded {len(loadable)} tensors from {sam3_checkpoint_path} (ratio={load_ratio:.2%})")
            if self.min_backbone_load_ratio > 0 and load_ratio < self.min_backbone_load_ratio:
                print(f"[SAM3] warning: load ratio below threshold ({self.min_backbone_load_ratio:.2%}).")
        else:
            print("[SAM3] warning: checkpoint path not found, using current init")

        for p in self.sam.parameters():
            p.requires_grad = False

        wrapped = [ConditionedAdapter(block, dino_dim=self.dino_dim, bottleneck=adapter_bottleneck) for block in self.sam.blocks]
        if isinstance(self.sam.blocks, nn.ModuleList):
            self.sam.blocks = nn.ModuleList(wrapped)
        else:
            self.sam.blocks = nn.Sequential(*wrapped)

        if len(set(self.sam_stage_ids)) != len(self.sam_stage_ids):
            raise ValueError("sam_stage_ids must be unique.")
        num_sam_blocks = len(self.sam.blocks)
        invalid_stage_ids = [sid for sid in self.sam_stage_ids if sid < 0 or sid >= num_sam_blocks]
        if invalid_stage_ids:
            raise ValueError(f"sam_stage_ids out of range [0, {num_sam_blocks - 1}]: {invalid_stage_ids}")

        self._sam_feats: Dict[int, torch.Tensor] = {}
        for sid in self.sam_stage_ids:
            self.sam.blocks[sid].register_forward_hook(self._make_hook(self._sam_feats, sid))

        # ---------- Align + reduce ----------
        self.sam_align = nn.ModuleList([nn.Conv2d(1024, 1024, 1) for _ in range(4)])
        self.sam_reduce = nn.ModuleList([nn.Conv2d(1024, self.fuse_dim, 1) for _ in range(4)])

        if self.use_clustering:
            self.region_fusion = nn.ModuleList([
                RegionFusionModule(self.fuse_dim, self.num_prototypes, self.fuse_dim) for _ in range(4)
            ])
            self.cluster_avg_pool_proj = nn.Sequential(
                nn.Linear(self.dino_dim, self.fuse_dim),
                nn.LayerNorm(self.fuse_dim),
                nn.GELU(),
            )
            self.prototype_pool = PrototypeAttentionPooling(self.dino_dim, self.fuse_dim) if self.use_prototype_attention_pooling else None
        else:
            self.region_fusion = None
            self.cluster_avg_pool_proj = None
            self.prototype_pool = None

        if self.use_cross_task_attention:
            self.grade_token_proj = nn.Linear(self.dino_dim, self.fuse_dim)
            self.cross_task_blocks = nn.ModuleList([
                BidirectionalCrossTaskAttention(dim=self.fuse_dim, num_heads=cross_attention_heads) for _ in range(4)
            ])
        else:
            self.grade_token_proj = None
            self.cross_task_blocks = None

        if self.use_dino_cond_refine:
            self.cond_fuse = nn.Sequential(
                nn.Linear(self.dino_dim * 3, self.dino_dim),
                nn.LayerNorm(self.dino_dim),
                nn.GELU(),
                nn.Linear(self.dino_dim, self.dino_dim),
            )
        else:
            self.cond_fuse = None

        dino_pool_hidden = max(64, self.dino_dim // 4)
        self.dino_cond_pool = nn.Sequential(
            nn.Linear(self.dino_dim, dino_pool_hidden),
            nn.GELU(),
            nn.Linear(dino_pool_hidden, 1),
        )
        self.dino_cond_blend = nn.Sequential(
            nn.Linear(self.dino_dim * 2, self.dino_dim),
            nn.LayerNorm(self.dino_dim),
            nn.GELU(),
        )

        if self.use_leaf_proxy_head:
            hidden = max(64, int(leaf_proxy_hidden))
            self.leaf_proxy_head = nn.Sequential(
                nn.LayerNorm(self.dino_dim),
                nn.Linear(self.dino_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.leaf_proxy_head = None

        if self.use_boundary_propagation:
            self.boundary_propagation = HierarchicalBoundaryPropagation(self.fuse_dim, num_levels=4)
            self.boundary_gate = nn.Conv2d(1, self.fuse_dim, 1)
        else:
            self.boundary_propagation = None
            self.boundary_gate = None

        # ---------- Decoder ----------
        self.up4 = UpBlock(self.fuse_dim * 2, self.fuse_dim)  # f4 + f3
        self.up3 = UpBlock(self.fuse_dim * 2, self.fuse_dim)  # + f2
        self.up2 = UpBlock(self.fuse_dim * 2, self.fuse_dim)  # + f1
        self.up1 = ConvBlock(self.fuse_dim, self.fuse_dim)    # refine only
        self.seg_head = nn.Conv2d(self.fuse_dim, self.num_classes, 1)

        grade_input_dim = self.fuse_dim * 4
        if self.use_clustering:
            grade_input_dim += self.fuse_dim
        if self.use_cross_task_attention:
            grade_input_dim += self.fuse_dim
        if self.use_boundary_propagation:
            grade_input_dim += self.fuse_dim

        self.grade_head = nn.Sequential(
            nn.Linear(grade_input_dim, self.fuse_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.fuse_dim * 2, self.num_grade_levels - 1),
        )
        if self.compare_last_layer_head:
            self.grade_head_last = nn.Sequential(
                nn.Linear(grade_input_dim, self.fuse_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.fuse_dim * 2, self.num_grade_levels - 1),
            )
        else:
            self.grade_head_last = None

        if self.use_consistency_correction:
            self.consistency_corrector = SegGradeConsistencyCorrector(
                grade_values_norm=self.grade_values_norm,
                grade_bounds=self.grade_bounds,
                temperature=consistency_temperature,
            )
        else:
            self.consistency_corrector = None

        self.sam.eval()
        self.dino.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hook(store: Dict[int, torch.Tensor], idx: int):
        def hook(_module, _input, output):
            store[idx] = output
        return hook

    @staticmethod
    def _anomaly_disease_gate(
        anomaly_w: torch.Tensor,
        leaf_w: torch.Tensor,
    ) -> torch.Tensor:
        """Soft-blend anomaly_w → leaf_w when anomaly is spatially diffuse
        (healthy leaf), and keep anomaly_w when it is concentrated (disease).

        Concentration ratio r = max / mean:
          * r ≈ 1  → uniform / diffuse  → healthy leaf  → use leaf_w
          * r >> 1 → peaked / localized → disease       → use anomaly_w

        Mapping: disease_gate = log(r) / log(N)  ∈ [0, 1]
        No additional parameters are introduced.
        """
        N = anomaly_w.shape[1]
        if N <= 1:
            return anomaly_w

        a_max  = anomaly_w.max(dim=1).values              # [B]
        a_mean = anomaly_w.mean(dim=1).clamp_min(1e-8)    # [B]
        # concentration ∈ [1, N]: 1 = perfectly uniform, N = single-token spike
        concentration = (a_max / a_mean).clamp(1.0, float(N))

        # log-scale normalisation → gate ∈ [0, 1]
        log_N = math.log(float(N))
        disease_gate = (torch.log(concentration) / log_N).clamp(0.0, 1.0)  # [B]

        alpha = disease_gate.unsqueeze(1)                  # [B, 1]
        return alpha * anomaly_w + (1.0 - alpha) * leaf_w

    def train(self, mode: bool = True):
        super().train(mode)

        # Keep frozen backbones in eval
        self.sam.eval()
        self.dino.eval()

        if mode:
            # Enable LoRA dropout if used
            for m in self.modules():
                if isinstance(m, LoRAQKV):
                    m.train()

            # Enable adapter heads; keep base block eval
            for m in self.modules():
                if isinstance(m, ConditionedAdapter):
                    m.prompt.train()
                    m.cond.train()
                    m.cond_spatial.train()
                    m.block.eval()

        return self

    def _dino_forward_features(self, x: torch.Tensor):
        if hasattr(self.dino, "forward_features"):
            return self.dino.forward_features(x)
        return self.dino(x)

    def _resolve_dino_layer_index(self, layer_idx: int) -> int:
        if self.dino_num_blocks <= 0:
            return -1
        idx = int(layer_idx)
        if idx < 0:
            idx = self.dino_num_blocks + idx
        if idx < 0 or idx >= self.dino_num_blocks:
            raise ValueError(
                f"dino_intermediate_layer out of range: {layer_idx}, "
                f"valid=[-{self.dino_num_blocks}, {self.dino_num_blocks - 1}]"
            )
        return idx

    def _extract_dino_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if hasattr(self.dino, "get_intermediate_layers") and self.dino_num_blocks > 0:
            last_idx = self.dino_num_blocks - 1
            requested = sorted({self.dino_intermediate_layer, last_idx})
            inter_outputs = None
            errors: List[str] = []

            try:
                out = self.dino.get_intermediate_layers(
                    x,
                    n=requested,
                    reshape=False,
                    return_class_token=True,
                    norm=True,
                )
                if len(out) != len(requested):
                    raise RuntimeError(
                        f"len(out)={len(out)} does not match requested={requested} with list n={requested}."
                    )
                inter_outputs = out
            except Exception as exc:
                errors.append(f"list-n failed: {type(exc).__name__}: {exc}")

            if inter_outputs is None:
                span = requested[-1] - requested[0] + 1
                try:
                    out = self.dino.get_intermediate_layers(
                        x,
                        n=span,
                        reshape=False,
                        return_class_token=True,
                        norm=True,
                    )
                    if len(out) != span:
                        raise RuntimeError(f"len(out)={len(out)} does not match span={span}")
                    inter_outputs = [out[layer_id - requested[0]] for layer_id in requested]
                except Exception as exc:
                    errors.append(f"last-n failed: {type(exc).__name__}: {exc}")

            if inter_outputs is None:
                raise RuntimeError(
                    "Failed to extract DINO intermediate layers. "
                    f"requested={requested}, model_blocks={self.dino_num_blocks}, errors={errors}"
                )

            token_map: Dict[int, torch.Tensor] = {}
            cls_map: Dict[int, Optional[torch.Tensor]] = {}
            for layer_id, layer_out in zip(requested, inter_outputs):
                if isinstance(layer_out, (tuple, list)):
                    patch = layer_out[0]
                    cls_token = layer_out[1] if len(layer_out) > 1 else None
                else:
                    patch = layer_out
                    cls_token = None
                token_map[layer_id] = _to_patch_token_sequence(patch).contiguous()
                cls_map[layer_id] = cls_token

            last_tokens = token_map[last_idx]
            last_cls = cls_map.get(last_idx)
            mid_tokens = token_map.get(self.dino_intermediate_layer, last_tokens)
            mid_cls = cls_map.get(self.dino_intermediate_layer, last_cls)
            return last_tokens, last_cls, mid_tokens, mid_cls

        dino_feats = self._dino_forward_features(x)
        patch_tokens = dino_feats.get("x_norm_patchtokens") if isinstance(dino_feats, dict) else dino_feats
        if patch_tokens is None:
            raise RuntimeError("DINO forward_features must provide patch tokens")
        patch_tokens = _to_patch_token_sequence(patch_tokens)
        expected_tokens, prefix_tokens, _patch, _grid = _infer_dino_layout(self.dino, self.low_res_size)
        patch_tokens = _strip_patch_tokens(patch_tokens, expected_tokens, prefix_tokens).contiguous()
        cls_token = dino_feats.get("x_norm_clstoken") if isinstance(dino_feats, dict) else None
        return patch_tokens, cls_token, patch_tokens, cls_token

    def _attn_pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.dino_cond_pool(tokens).squeeze(-1)   # [B,N]
        weights = F.softmax(logits, dim=1)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)

    def ratio_to_grade_index(self, ratio: torch.Tensor) -> torch.Tensor:
        idx = torch.full_like(ratio, len(self.grade_bounds) - 1, dtype=torch.long)
        for i, (lo, hi) in enumerate(self.grade_bounds):
            if i < len(self.grade_bounds) - 1:
                mask = (ratio >= lo) & (ratio < hi)
            else:
                mask = (ratio >= lo) & (ratio <= hi)
            idx = torch.where(mask, torch.tensor(i, device=ratio.device, dtype=torch.long), idx)
        return idx

    def _set_condition(self, cond: torch.Tensor, cond_map: Optional[torch.Tensor] = None):
        for blk in self.sam.blocks:
            if hasattr(blk, "set_condition_map"):
                blk.set_condition_map(cond, cond_map)
            elif hasattr(blk, "set_condition"):
                blk.set_condition(cond)

    def _disease_leaf_probs(self, seg_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seg_prob = F.softmax(seg_logits, dim=1)
        if self.leaf_class_id >= seg_prob.shape[1]:
            raise RuntimeError(f"leaf_class_id={self.leaf_class_id} out of range")
        disease_prob = seg_prob[:, self.disease_class_ids].sum(dim=1)
        leaf_total_prob = seg_prob[:, self.leaf_class_id] + disease_prob
        return disease_prob, leaf_total_prob

    def _decode_from_fused(
        self,
        fused_feats: Sequence[torch.Tensor],
        out_size: Tuple[int, int],
        apply_boundary: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        fused = list(fused_feats)
        boundary_logits: List[torch.Tensor] = []
        boundary_fused: Optional[torch.Tensor] = None
        boundary_context: Optional[torch.Tensor] = None

        if apply_boundary and self.use_boundary_propagation and self.boundary_propagation is not None:
            fused, boundary_logits, boundary_fused, boundary_context = self.boundary_propagation(fused)

        f1, f2, f3, f4 = fused

        y = self.up4(f4, f3)   # 1/32 -> 1/16
        y = self.up3(y, f2)    # 1/16 -> 1/8
        y = self.up2(y, f1)    # 1/8  -> 1/4
        y = self.up1(y)        # refine

        if boundary_fused is not None and self.boundary_gate is not None:
            gate = F.interpolate(boundary_fused, size=y.shape[2:], mode="bilinear", align_corners=False)
            y = y + self.boundary_gate(torch.sigmoid(gate))

        seg_logits = self.seg_head(y)
        seg_logits = F.interpolate(seg_logits, size=out_size, mode="bilinear", align_corners=False)
        return seg_logits, [f1, f2, f3, f4], boundary_logits, boundary_fused, boundary_context

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_aux: bool = True):
        self._sam_feats.clear()
        try:
            return self._forward_impl(x, return_aux)
        finally:
            self._sam_feats.clear()

    def _forward_impl(self, x: torch.Tensor, return_aux: bool = True):
        bsz, _, h, w = x.shape
        target_sizes = [
            (_ceil_div(h, 4), _ceil_div(w, 4)),
            (_ceil_div(h, 8), _ceil_div(w, 8)),
            (_ceil_div(h, 16), _ceil_div(w, 16)),
            (_ceil_div(h, 32), _ceil_div(w, 32)),
        ]

        # ---------- DINO ----------
        x_low = F.interpolate(x, size=(self.low_res_size, self.low_res_size), mode="bilinear", align_corners=False)
        if self.use_dino_lora:
            patch_tokens, cls_token_last, patch_tokens_mid, cls_token_mid = self._extract_dino_tokens(x_low)
        else:
            with torch.no_grad():
                patch_tokens, cls_token_last, patch_tokens_mid, cls_token_mid = self._extract_dino_tokens(x_low)

        # Strip prefix tokens for both last/mid
        expected_tokens, prefix_tokens, _patch, _grid = _infer_dino_layout(self.dino, self.low_res_size)
        patch_tokens = _strip_patch_tokens(patch_tokens, expected_tokens, prefix_tokens).contiguous()
        patch_tokens_mid = _strip_patch_tokens(patch_tokens_mid, expected_tokens, prefix_tokens).contiguous()

        token_len = patch_tokens.shape[1]
        grid = int(token_len ** 0.5)
        if grid * grid != token_len:
            raise RuntimeError(f"DINO patch token length {token_len} is not a perfect square.")
        patch_tokens = patch_tokens.contiguous()
        patch_tokens_mid = patch_tokens_mid.contiguous()

        # Attention pooling for conditioning
        dino_cond_last = self._attn_pool_tokens(patch_tokens)
        dino_cond_mid = self._attn_pool_tokens(patch_tokens_mid)
        dino_cond = self.dino_cond_blend(torch.cat([dino_cond_mid, dino_cond_last], dim=1))

        # ---------- Leaf proxy + anomaly ----------
        dino_cond_refined = dino_cond

        leaf_proxy_logits = None
        leaf_proxy_prob = None
        leaf_proxy_grid = None
        leaf_proxy_hw = None

        anomaly_grid = None
        leaf_w = None
        anomaly_w = None

        roi_map_grid = None

        if self.use_leaf_proxy_head and self.leaf_proxy_head is not None:
            tokens_flat = patch_tokens.reshape(-1, self.dino_dim)
            leaf_proxy_logits = self.leaf_proxy_head(tokens_flat).view(bsz, -1, 1)
            leaf_proxy_prob = torch.sigmoid(leaf_proxy_logits).squeeze(-1)  # [B,N]
            leaf_proxy_grid = leaf_proxy_prob.view(bsz, grid, grid)

            leaf_proxy_hw = F.interpolate(
                leaf_proxy_grid.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            leaf_w = leaf_proxy_prob.clamp_min(1e-6)
            tokens_norm = F.normalize(patch_tokens, dim=-1)

            def _weighted_mean(weights: torch.Tensor) -> torch.Tensor:
                denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                return (tokens_norm * weights.unsqueeze(-1)).sum(dim=1) / denom

            leaf_cond = _weighted_mean(leaf_w)

            if self.roi_use_anomaly:
                leaf_center = F.normalize(leaf_cond, dim=-1)
                sim = (tokens_norm * leaf_center.unsqueeze(1)).sum(dim=-1)
                anomaly = (1.0 - sim).clamp_min(0.0) * leaf_w
                a_min = anomaly.min(dim=1, keepdim=True).values
                a_max = anomaly.max(dim=1, keepdim=True).values
                anomaly = (anomaly - a_min) / (a_max - a_min + 1e-6)
                anomaly_w = anomaly.clamp_min(1e-6)
                anomaly_grid = anomaly.view(bsz, grid, grid)

            if self.cond_fuse is not None:
                # Use concentration-gated anomaly weight to avoid healthy-leaf interference
                if anomaly_w is not None and leaf_w is not None:
                    w_for_anom = self._anomaly_disease_gate(anomaly_w, leaf_w)
                else:
                    w_for_anom = anomaly_w if anomaly_w is not None else leaf_w
                anom_cond = _weighted_mean(w_for_anom)
                dino_cond_refined = self.cond_fuse(torch.cat([dino_cond, leaf_cond, anom_cond], dim=1))

            # Simple ROI grid (leaf * anomaly)
            roi_grid = leaf_proxy_grid
            if self.roi_use_anomaly and anomaly_grid is not None:
                roi_grid = roi_grid * anomaly_grid
            roi_grid = (self.roi_center_weight * roi_grid).clamp(0.0, 1.0)
            roi_map_grid = roi_grid.unsqueeze(1)

        # ---------- Spatial cond map aligned to SAM token grid ----------
        cond_map_aligned: Optional[torch.Tensor] = None
        if patch_tokens_mid.dim() == 3:
            mid_grid = int(patch_tokens_mid.shape[1] ** 0.5)
            if mid_grid * mid_grid != patch_tokens_mid.shape[1]:
                raise RuntimeError("DINO mid-layer tokens are not a perfect square.")
            cond_map_aligned = patch_tokens_mid.view(bsz, mid_grid, mid_grid, self.dino_dim)

            sam_grid = max(1, int(self.high_res_size) // int(self.sam_patch_size))
            if mid_grid != sam_grid:
                cond_map_aligned = F.interpolate(
                    cond_map_aligned.permute(0, 3, 1, 2),
                    size=(sam_grid, sam_grid),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()
            else:
                cond_map_aligned = cond_map_aligned.contiguous()
        elif patch_tokens_mid.dim() == 4:
            cond_map_aligned = patch_tokens_mid

        # Set conditioning for SAM
        self._set_condition(dino_cond_refined, cond_map=cond_map_aligned)

        # ---------- SAM ----------
        x_high = F.interpolate(x, size=(self.high_res_size, self.high_res_size), mode="bilinear", align_corners=False)
        _ = self.sam(x_high)

        sam_maps: List[torch.Tensor] = []
        for sid, size in zip(self.sam_stage_ids, target_sizes):
            feat = self._sam_feats.get(sid)
            if feat is None:
                raise RuntimeError(f"SAM stage {sid} did not produce feature")
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = _tokens_to_map(feat, bsz, embed_dim=1024)
            feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            sam_maps.append(feat)

        base_fused: List[torch.Tensor] = []
        for i in range(4):
            f = self.sam_align[i](sam_maps[i])
            f = self.sam_reduce[i](f)
            base_fused.append(f)

        # ---------- Clustering + fusion ----------
        cluster_results = None
        refined_fused = list(base_fused)

        proto_attn = None
        pooled_cluster_feat = None

        if self.use_clustering and self.clustering is not None and self.region_fusion is not None:
            if self.cluster_token_source == "mid":
                cluster_tokens = patch_tokens_mid
            elif self.cluster_token_source == "last":
                cluster_tokens = patch_tokens
            else:
                cluster_tokens = 0.5 * (patch_tokens_mid + patch_tokens)

            # Determine cluster ROI weight with concentration-aware gating
            cluster_roi_weight = None
            if self.cluster_roi_source == "anomaly":
                cluster_roi_weight = anomaly_w
            elif self.cluster_roi_source == "leaf":
                cluster_roi_weight = leaf_w
            elif self.cluster_roi_source == "anomaly_then_leaf":
                if anomaly_w is not None and leaf_w is not None:
                    # On healthy leaves, anomaly_w is diffuse pseudo-noise; gate towards leaf_w.
                    # On diseased leaves, anomaly_w is concentrated; keep it.
                    cluster_roi_weight = self._anomaly_disease_gate(anomaly_w, leaf_w)
                else:
                    cluster_roi_weight = anomaly_w if anomaly_w is not None else leaf_w

            if not self.use_soft_roi_cluster:
                cluster_roi_weight = None

            cluster_results = self.clustering(cluster_tokens, grid, roi_weight=cluster_roi_weight)

            # Use weighted map if ROI clustering enabled and ROI weights exist
            if self.use_soft_roi_cluster and cluster_roi_weight is not None:
                cluster_map = cluster_results["cluster_map_weighted"]
            else:
                cluster_map = cluster_results["cluster_map"]

            for i in range(4):
                refined_fused[i] = self.region_fusion[i](
                    refined_fused[i],
                    cluster_map,
                    roi_gate=roi_map_grid if (self.use_soft_roi_cluster and self.use_roi_gate_in_fusion) else None,
                )

            if self.prototype_pool is not None:
                pooled_cluster_feat, proto_attn = self.prototype_pool(
                    cluster_results["cluster_feat"],
                    cluster_mass=cluster_results.get("cluster_mass"),
                )
            elif self.cluster_avg_pool_proj is not None:
                cluster_avg = cluster_results["cluster_feat"].mean(dim=1)
                pooled_cluster_feat = self.cluster_avg_pool_proj(cluster_avg)

        # ---------- Cross-task attention ----------
        grade_token = None
        if self.use_cross_task_attention and self.cross_task_blocks is not None and self.grade_token_proj is not None:
            grade_token = self.grade_token_proj(dino_cond_refined)
            # from coarse to fine
            for i in range(3, -1, -1):
                refined_fused[i], grade_token = self.cross_task_blocks[i](refined_fused[i], grade_token)

        # ---------- Decode ----------
        seg_logits, decoded_feats, boundary_logits, boundary_fused, boundary_context = self._decode_from_fused(
            fused_feats=refined_fused,
            out_size=(h, w),
            apply_boundary=True,
        )
        f1, f2, f3, f4 = decoded_feats

        if not return_aux:
            return seg_logits

        disease_prob, leaf_total_prob = self._disease_leaf_probs(seg_logits)
        disease_ratio = disease_prob.sum(dim=(1, 2)) / (leaf_total_prob.sum(dim=(1, 2)) + 1e-6)

        # ---------- Grade feature pooling ----------
        if self.grade_use_disease_weighted_pool:
            weight = disease_prob * leaf_total_prob
            if self.grade_pool_detach:
                weight = weight.detach()

            pooled = []
            for feat in (f1, f2, f3, f4):
                w_map = F.interpolate(weight.unsqueeze(1), size=feat.shape[2:], mode="bilinear", align_corners=False)
                w_map = w_map.clamp_min(1e-6)
                num = (feat * w_map).sum(dim=(2, 3))
                den = w_map.sum(dim=(2, 3)).clamp_min(1e-6)
                pooled.append(num / den)
            grade_feat_shared = torch.cat(pooled, dim=1)
        else:
            pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in (f1, f2, f3, f4)]
            grade_feat_shared = torch.cat(pooled, dim=1)

        if pooled_cluster_feat is not None:
            grade_feat_shared = torch.cat([grade_feat_shared, pooled_cluster_feat], dim=1)
        if boundary_context is not None:
            boundary_vec = F.adaptive_avg_pool2d(boundary_context, 1).flatten(1)
            grade_feat_shared = torch.cat([grade_feat_shared, boundary_vec], dim=1)

        grade_feat = grade_feat_shared
        if grade_token is not None:
            grade_feat = torch.cat([grade_feat, grade_token], dim=1)

        # ---------- Grade heads ----------
        grade_logits = self.grade_head(grade_feat)
        grade_probs_head = _ordinal_probs_from_logits(grade_logits)
        grade_score_head = (grade_probs_head * self.grade_values_norm.unsqueeze(0)).sum(dim=1)

        grade_logits_last = None
        grade_probs_last = None
        grade_score_last = None
        if self.compare_last_layer_head and self.grade_head_last is not None:
            shared_for_last = grade_feat_shared.detach() if self.detach_shared_for_last_head else grade_feat_shared
            grade_feat_last = shared_for_last
            if self.use_cross_task_attention and self.grade_token_proj is not None:
                grade_token_last = self.grade_token_proj(dino_cond_last)
                grade_feat_last = torch.cat([grade_feat_last, grade_token_last], dim=1)
            grade_logits_last = self.grade_head_last(grade_feat_last)
            grade_probs_last = _ordinal_probs_from_logits(grade_logits_last)
            grade_score_last = (grade_probs_last * self.grade_values_norm.unsqueeze(0)).sum(dim=1)

        # Threshold grade from ratio
        grade_from_ratio = self.ratio_to_grade_index(disease_ratio)
        ratio_grade_probs = F.one_hot(grade_from_ratio, num_classes=self.num_grade_levels).float()
        ratio_grade_score = (ratio_grade_probs * self.grade_values_norm.unsqueeze(0)).sum(dim=1)

        inconsistency_score = (grade_score_head - ratio_grade_score).abs()
        consistency_weight_head = torch.ones_like(inconsistency_score)

        grade_probs = grade_probs_head
        grade_score = grade_score_head

        if self.use_consistency_correction and self.consistency_corrector is not None:
            cons = self.consistency_corrector(grade_probs_head, grade_score_head, disease_ratio)
            ratio_grade_probs = cons["ratio_grade_probs"]
            ratio_grade_score = cons["ratio_grade_score"]
            inconsistency_score = cons["inconsistency_score"]
            consistency_weight_head = cons["consistency_weight_head"]
            grade_probs = cons["corrected_probs"]
            grade_score = cons["corrected_score"]

        aux: Dict[str, torch.Tensor] = {
            "disease_ratio": disease_ratio,
            "leaf_total_prob": leaf_total_prob,
            "disease_prob": disease_prob,
            "grade_logits": grade_logits,
            "grade_probs_head": grade_probs_head,
            "grade_score_head": grade_score_head,
            "grade_probs": grade_probs,
            "grade_score": grade_score,
            "grade_from_ratio": grade_from_ratio,
            "ratio_grade_probs": ratio_grade_probs,
            "ratio_grade_score": ratio_grade_score,
            "inconsistency_score": inconsistency_score,
            "consistency_weight_head": consistency_weight_head,
            "grade_pred_idx": torch.argmax(grade_probs, dim=1),
            "dino_patch_tokens": patch_tokens,
            "dino_grid_size": grid,
        }

        if grade_logits_last is not None and grade_probs_last is not None and grade_score_last is not None:
            aux["grade_logits_last"] = grade_logits_last
            aux["grade_probs_last"] = grade_probs_last
            aux["grade_score_last"] = grade_score_last

        if leaf_proxy_logits is not None:
            aux["leaf_proxy_logits"] = leaf_proxy_logits
        if leaf_proxy_prob is not None:
            aux["leaf_proxy_prob"] = leaf_proxy_prob
        if leaf_proxy_grid is not None:
            aux["leaf_proxy_grid"] = leaf_proxy_grid
        if leaf_proxy_hw is not None:
            aux["leaf_proxy_hw"] = leaf_proxy_hw
        if anomaly_grid is not None:
            aux["anomaly_grid"] = anomaly_grid

        if boundary_logits:
            aux["boundary_logits"] = boundary_logits
        if boundary_fused is not None:
            aux["boundary_fused"] = boundary_fused

        if self.use_clustering and cluster_results is not None and self.clustering is not None:
            aux.update({
                "cluster_map": cluster_results["cluster_map"],
                "cluster_map_weighted": cluster_results.get("cluster_map_weighted"),
                "cluster_feat": cluster_results["cluster_feat"],
                "cluster_assignment": cluster_results["cluster_assignment"],
                "proto_similarity": cluster_results["proto_similarity"],
                "cluster_probs": cluster_results["cluster_probs"],
                "cluster_mass": cluster_results["cluster_mass"],
                "cluster_token_weight": cluster_results.get("cluster_token_weight") if cluster_results.get("cluster_token_weight") is not None else None,
                "prototypes": self.clustering.prototypes,
                "cluster_proto_attn": proto_attn if proto_attn is not None else None,
                "proto_diversity_loss": self.clustering.prototype_diversity_loss(),
            })
            if roi_map_grid is not None:
                aux["cluster_roi_map"] = roi_map_grid

        return seg_logits, aux

    def get_trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a state_dict containing all parameters that require grad."""
        full_sd = self.state_dict()
        trainable_names = {n for n, p in self.named_parameters() if p.requires_grad}
        prefixes = {n.rsplit(".", 1)[0] for n in trainable_names}

        keep: Dict[str, torch.Tensor] = {}
        for k, v in full_sd.items():
            if k in trainable_names:
                keep[k] = v
            else:
                pref = k.rsplit(".", 1)[0]
                if pref in prefixes:
                    keep[k] = v

        # Keep key buffers used in inference
        if "grade_values_norm" in full_sd:
            keep["grade_values_norm"] = full_sd["grade_values_norm"]
        return keep


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = DMNetWithClustering(
        sam3_checkpoint_path="/mnt/nas/szyz/home/jwwang/DINOv3SAM3-UNet/load/sam3.pt",
        dinov3_weight_path="/mnt/nas/szyz/home/jwwang/DINOv3SAM3-UNet/ckpt/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        dinov3_local_path="./dinov3",
        dinov3_model_name="dinov3_vitl16",
        num_prototypes=6,
        use_clustering=True,
    ).to(dev)

    model.eval()
    dummy = torch.randn(2, 3, 512, 512, device=dev)
    with torch.no_grad():
        logits, aux = model(dummy, return_aux=True)

    print("=" * 60)
    print("模型输出:")
    print("=" * 60)
    print(f"分割 logits: {logits.shape}")
    print(f"病害比例: {aux['disease_ratio'].shape}")
    print(f"DINO tokens: {aux['dino_patch_tokens'].shape}")

    if "cluster_map" in aux:
        print("\n聚类相关输出:")
        print(f"  聚类图: {aux['cluster_map'].shape}")
        print(f"  聚类特征: {aux['cluster_feat'].shape}")
        print(f"  聚类分配: {aux['cluster_assignment'].shape}")
        print(f"  原型向量: {aux['prototypes'].shape}")
        print(f"  相似度矩阵: {aux['proto_similarity'].shape}")