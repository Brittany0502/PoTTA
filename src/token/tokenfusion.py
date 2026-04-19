import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SaliencyAttnPool(nn.Module):


    def __init__(self, dim, use_local_contrast=True):
        super().__init__()
        self.use_local_contrast = use_local_contrast
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)

    @torch.no_grad()
    def _saliency_score(self, x):  # x: [B, T, C]
        # 通道能量
        energy = x.pow(2).mean(-1)  # [B, T]
        if not self.use_local_contrast:
            return energy
        # 局部对比（邻域均值差异）——用 unfolding 近似
        # 需要 H, W；从 T 恢复
        return energy

    def forward(self, feat_2d):
        """
        feat_2d: [B, C, H, W]
        returns:
          v_fg: [B, C]
          attn_map: [B, 1, H, W]
        """
        B, C, H, W = feat_2d.shape
        tokens = feat_2d.flatten(2).transpose(1, 2)  # [B, T, C], T=H*W


        with torch.no_grad():
            sal = tokens.pow(2).mean(-1)  # [B, T]

        return tokens, sal

    def attn_pool(self, tokens, mask_topk):
        """
        tokens: [B, T, C]
        mask_topk: [B, T] bool
        """
        B, T, C = tokens.shape

        q_sal = []
        for b in range(B):
            sel = tokens[b][mask_topk[b]]  # [K, C]
            if sel.numel() == 0:
                sel = tokens[b]
            q_sal.append(sel.mean(0, keepdim=True))
        q_sal = torch.stack(q_sal, 0)  # [B, 1, C]

        Q = self.Wq(q_sal)  # [B, 1, C]
        K = self.Wk(tokens)  # [B, T, C]
        V = self.Wv(tokens)  # [B, T, C]

        attn = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(K.shape[-1])  # [B, 1, T]
        attn = attn.softmax(-1)  # [B, 1, T]
        v_fg = torch.matmul(attn, V).squeeze(1)  # [B, C]
        return v_fg, attn

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x):
        # [B,C,H,W]
        u = x.mean(dim=(2,3), keepdim=True)
        s = (x - u).pow(2).mean(dim=(2,3), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:,None,None]*x + self.bias[:,None,None]

class AttentionPool2dLite(nn.Module):

    def __init__(self, spatial_hw: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(spatial_hw*spatial_hw + 1, embed_dim) / math.sqrt(embed_dim))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, feat_2d: torch.Tensor):
        # feat_2d: [B,C,H,W]
        B, C, H, W = feat_2d.shape
        T = H*W
        x = feat_2d.flatten(2).transpose(1, 2)           # [B,T,C]
        cls0 = x.mean(dim=1, keepdim=True)               # [B,1,C] 作为全局query初始化
        x = torch.cat([cls0, x], dim=1)                  # [B,1+T,C]
        x = x + self.pos[None, :, :].to(x.dtype)

        # 只对 query=cls 做 MHA（和参考实现一致的“单查询”聚合）
        q = self.q_proj(x[:, :1, :])                     # [B,1,C]
        k = self.k_proj(x)                               # [B,1+T,C]
        v = self.v_proj(x)

        # 变多头
        def split_heads(t):
            return t.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        qh, kh, vh = map(split_heads, (q, k, v))         # qh:[B,h,1,d], kh/vh:[B,h,1+T,d]

        attn = (qh @ kh.transpose(-2, -1)) / math.sqrt(C // self.num_heads)   # [B,h,1,1+T]
        attn = attn.softmax(dim=-1)
        pooled = (attn @ vh)                             # [B,h,1,d]
        pooled = pooled.transpose(1,2).reshape(B,1,C)    # [B,1,C]
        pooled = self.out(pooled).squeeze(1)             # [B,C]


        attn_sp = attn.mean(dim=1).squeeze(1)[:, 1:]     # [B,T]
        attn_sp = attn_sp.view(B, 1, H, W)               # [B,1,H,W]
        attn_sp = attn_sp / (attn_sp.amax(dim=(2,3), keepdim=True) + 1e-6)
        return pooled, attn_sp


class AttentionPool2dLiteV2(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self._pos_cache = {}  # key=int(1+T) -> nn.Parameter

    def _get_pos(self, length: int, dtype, device):
        if length not in self._pos_cache:
            p = nn.Parameter(torch.randn(length, self.embed_dim) / math.sqrt(self.embed_dim))
            self._pos_cache[length] = p
            self.register_parameter(f"pos_{length}", self._pos_cache[length])
        return self._pos_cache[length].to(device=device, dtype=dtype)

    def forward(self, feat_2d: torch.Tensor):
        B, C, H, W = feat_2d.shape
        x = feat_2d.flatten(2).transpose(1, 2)  # [B,T,C]
        cls0 = x.mean(dim=1, keepdim=True)
        x = torch.cat([cls0, x], dim=1)         # [B,1+T,C]
        pos = self._get_pos(x.shape[1], x.dtype, x.device)
        x = x + pos[None, :, :]

        q = self.q_proj(x[:, :1, :])
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(t):
            return t.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        qh, kh, vh = map(split_heads, (q, k, v))
        attn = (qh @ kh.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        attn = attn.softmax(dim=-1)
        pooled = (attn @ vh).transpose(1,2).reshape(B,1,C)
        pooled = self.out(pooled).squeeze(1)    # [B,C]

        attn_sp = attn.mean(dim=1).squeeze(1)[:, 1:].view(B, 1, H, W)
        attn_sp = attn_sp / (attn_sp.amax(dim=(2,3), keepdim=True) + 1e-6)
        return pooled, attn_sp

class RegionAwareTokenFusionV2(nn.Module):

    def __init__(self, dim, hw: int, heads: int = 4,
                 topk_start=0.05, topk_end=0.15,
                 alpha_max=0.35,
                 gamma=0.5,
                 warmup_steps=1500):
        super().__init__()
        self.dim = dim
        self.gamma = gamma
        self.alpha_max = alpha_max
        self.topk_start = topk_start
        self.topk_end = topk_end
        self.warmup_steps = warmup_steps

        self.pool = AttentionPool2dLite(hw, dim, heads)
        self.pre_norm = LayerNorm2d(dim)
        self.post_norm = LayerNorm2d(dim)

        self.to_gate = nn.Sequential(
            nn.Conv2d(dim, dim//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//4, dim, 1, bias=True),
            nn.Sigmoid()
        )
        self.register_buffer("global_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def _schedule(self):

        t = float(min(int(self.global_step.item()), self.warmup_steps))
        ratio = 0.5 * (1 - math.cos(math.pi * t / self.warmup_steps))
        alpha = ratio * self.alpha_max
        topk = self.topk_start + (self.topk_end - self.topk_start) * ratio
        return alpha, topk

    @torch.no_grad()
    def _saliency_energy(self, feat_2d):

        B, C, H, W = feat_2d.shape
        tokens = feat_2d.flatten(2).transpose(1, 2) # [B,T,C]
        sal = tokens.pow(2).mean(-1)                # [B,T]
        return sal

    def forward(self, feat_2d):
        B, C, H, W = feat_2d.shape
        self.global_step += 1
        alpha, topk_ratio = self._schedule()

        x = self.pre_norm(feat_2d)
        pooled, attn_map = self.pool(x)             # pooled:[B,C], attn:[B,1,H,W]


        with torch.no_grad():
            sal = self._saliency_energy(x)          # [B,T]
            T = sal.shape[1]
            k = max(1, int(T * topk_ratio))
            _, idx = torch.topk(sal, k, dim=1)
            mask = torch.zeros_like(sal, dtype=torch.bool)
            mask.scatter_(1, idx, True)             # [B,T]

        tokens = x.flatten(2).transpose(1,2)        # [B,T,C]
        refine = []
        for b in range(B):
            sel = tokens[b][mask[b]]
            if sel.numel() == 0: sel = tokens[b]
            refine.append(sel.mean(0, keepdim=True))
        refine = torch.cat(refine, 0)               # [B,C]
        v_fg = 0.8*pooled + 0.2*refine


        v_glb = tokens.mean(1)                      # [B,C]
        v_fused = self.gamma * v_glb + (1 - self.gamma) * v_fg
        gate = self.to_gate(v_fused.view(B,C,1,1))  # [B,C,1,1]


        attn_detached = attn_map.detach()
        fused_feat = x * gate * (1.0 + attn_detached)


        out = feat_2d + alpha * (self.post_norm(fused_feat) - feat_2d)
        return out.contiguous(), attn_map

def _boxes_to_gaussian_heatmap(boxes_xyxy, scores, H, W, img_hw, sigma=2.0):

    device = boxes_xyxy.device
    heat = torch.zeros((H, W), device=device, dtype=torch.float32)

    if boxes_xyxy.numel() == 0:
        return heat[None:None, ...]  # [1,H,W]

    img_h, img_w = img_hw
    scale_y, scale_x = H / float(img_h), W / float(img_w)

    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5 * scale_x
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5 * scale_y
    bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * scale_x
    bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) * scale_y
    sigmas = sigma * torch.clamp_min((bw * bh).sqrt() / 6.0, 1.0)

    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    for i in range(cx.numel()):
        g = torch.exp(-0.5 * (((xx - cx[i]) / sigmas[i])**2 + ((yy - cy[i]) / sigmas[i])**2))
        heat = torch.maximum(heat, scores[i] * g)


    m = heat.amax()
    if m > 0:
        heat = heat / m
    return heat[None:None, ...]  # [1,1,H,W]

def _binary_kl(p, q, eps=1e-6):

    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return (p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log()))