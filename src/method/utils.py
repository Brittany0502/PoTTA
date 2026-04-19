import torch

# =========================
# Discrepancy-Amplifying Adapter (DAA) utilities
# =========================
def _find_adapter_backbone(model: torch.nn.Module):
    """Find backbone module that contains adapters_mod0 and enable_adapters flag."""
    # common attribute
    for attr in ["backbone", "module"]:
        if hasattr(model, attr):
            cand = getattr(model, attr)
            if isinstance(cand, torch.nn.Module) and hasattr(cand, "adapters_mod0"):
                return cand
    # fallback: search modules
    for m in model.modules():
        if hasattr(m, "adapters_mod0") and hasattr(m, "enable_adapters"):
            return m
    return None

def _gap(feat: torch.Tensor) -> torch.Tensor:
    """Global average pool (B,C,H,W) -> (B,C)."""
    if feat.dim() == 4:
        return feat.mean(dim=(2, 3))
    return feat

def _daa_loss_from_feats(backbone: torch.nn.Module,
                         feats_no: list,
                         feats_ad: list,
                         beta_alpha: float = 1.0,
                         noise_std: float = 0.01,
                         temperature: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute:
      - preserve loss: MSE(DAA(r), r) on pooled features
      - discrepancy-amplify loss: InfoNCE between two augmented adapted unknown features,
        with negatives from adapted known features in batch (detached).
    """
    device = feats_ad[0].device
    # ---- preserve (known) ----
    preserve = 0.0
    for f_no, f_ad in zip(feats_no, feats_ad):
        preserve = preserve + F.mse_loss(_gap(f_ad), _gap(f_no))

    # ---- unknown simulation: mixup + gaussian noise on *original* features ----
    B = feats_no[0].shape[0]
    perm = torch.randperm(B, device=device)

    # sample mixup coefficient lambda ~ Beta(alpha, alpha)
    if beta_alpha <= 0:
        lam = torch.full((B, 1, 1, 1), 0.5, device=device)
    else:
        # torch.distributions.Beta supports vector sampling
        lam_scalar = torch.distributions.Beta(beta_alpha, beta_alpha).sample((B,)).to(device)
        lam = lam_scalar.view(B, 1, 1, 1)

    un_loss = 0.0
    for lvl, (f_no, f_ad) in enumerate(zip(feats_no, feats_ad)):
        f1 = f_no.detach()
        f2 = f_no.detach()[perm]
        r_tilde = lam * f1 + (1.0 - lam) * f2
        if noise_std > 0:
            r_tilde = r_tilde + noise_std * torch.randn_like(r_tilde)

        # pass through adapter ONLY for this level
        # --- FIX: map feat-level index -> backbone stage index ---
        if hasattr(backbone, "return_idx"):
            stage_idx = backbone.return_idx[lvl]
        else:
            stage_idx = lvl  # fallback
        adapter = backbone.adapters_mod0[stage_idx]
        u1 = adapter(r_tilde)
        # augmented view
        u2 = adapter(r_tilde + (noise_std * torch.randn_like(r_tilde) if noise_std > 0 else 0.0))

        z1 = F.normalize(_gap(u1), dim=-1)
        z2 = F.normalize(_gap(u2), dim=-1)

        # negatives: adapted known features (detached) in batch
        z_kn = F.normalize(_gap(f_ad.detach()), dim=-1)  # (B,C)

        # InfoNCE: pos = (z1·z2), negs = (z1·z_kn_j)
        pos = torch.sum(z1 * z2, dim=-1, keepdim=True) / temperature  # (B,1)
        neg = (z1 @ z_kn.t()) / temperature  # (B,B)
        # mask out diagonal? here z_kn are known from same batch; keep all as negatives
        logits = torch.cat([pos, neg], dim=1)  # (B, 1+B)
        labels = torch.zeros(B, dtype=torch.long, device=device)  # pos index=0
        un_loss = un_loss + F.cross_entropy(logits, labels)

    return preserve, un_loss

# =========== 度量学习 ========================
def extract_encoder_tokens_after_attn(model, layer_idx=-1):
    """
    从指定 encoder 层取 MHSA 输出 token 表示（不含 [cls]/query），尺寸 [B, Hs*Ws, C] 与 [B,Hs,Ws]。
    你若已有 _saliency_cache，可在 forward hook 里直接缓存；这里给一个接口约定。
    """
    # 假设你在 forward hook 里把该层的输出保存在 model._enc_tokens (B, C, Hs, Ws)
    feat = getattr(model, "_enc_tokens", None)  # [B,C,Hs,Ws]
    if feat is None:
        return None, None, None
    B, C, Hs, Ws = feat.shape
    tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, Hs*Ws, C]
    return tokens, Hs, Ws

def tokens_to_xy(tokens_idx, Hs, Ws, img_hw):
    """把第 idx 个token（平铺索引）映射回原图像素坐标 (cx, cy)。"""
    Himg, Wimg = img_hw
    yy = (tokens_idx // Ws).float() / max(Hs-1, 1)  # 0~1
    xx = (tokens_idx %  Ws).float() / max(Ws-1, 1)
    cx = xx * (Wimg-1)
    cy = yy * (Himg-1)
    return cx, cy

def point_in_boxes(cx, cy, boxes_xyxy):
    """判断点 (cx,cy) 是否落在任一 boxes 里。boxes:[K,4] 像素坐标。"""
    if boxes_xyxy.numel() == 0:
        return False
    x1y1 = boxes_xyxy[:, :2]
    x2y2 = boxes_xyxy[:, 2:]
    cond = (cx >= x1y1[:,0]) & (cy >= x1y1[:,1]) & (cx <= x2y2[:,0]) & (cy <= x2y2[:,1])
    return bool(cond.any().item())

def sample_salient_tokens(tokens, sal, mask_pos, mask_neg, k_pos=64, k_neg=64):
    """根据显著性与正/负掩码采样 token 索引。"""
    # sal: [B, N], mask_*: [B, N] bool
    pos_idx, neg_idx = [], []
    with torch.no_grad():
        sal_pos = sal.masked_fill(~mask_pos, -1e9)
        sal_neg = sal.masked_fill(~mask_neg, -1e9)
        # topk 可能不足，做安全处理
        kpos = min(k_pos, int((mask_pos.sum(dim=1).amax()).item()) if mask_pos.any() else 0)
        kneg = min(k_neg, int((mask_neg.sum(dim=1).amax()).item()) if mask_neg.any() else 0)
        if kpos > 0:
            pos_idx = torch.topk(sal_pos, kpos, dim=1).indices  # [B,kpos]
        if kneg > 0:
            neg_idx = torch.topk(sal_neg, kneg, dim=1).indices  # [B,kneg]
    return pos_idx, neg_idx

def info_nce_query_token_loss(q_vec, pos_tokens, neg_tokens, temperature=0.07):
    """
    q_vec:   [B, C]    选定的“锚点查询”（如与某 GT/伪标匹配的query向量的均值/EMA）
    pos_tokens: [B, Kp, C]
    neg_tokens: [B, Kn, C]
    """
    B, C = q_vec.shape
    q = F.normalize(q_vec, dim=-1)
    loss = torch.tensor(0.0, device=q_vec.device)
    if pos_tokens.numel() == 0:
        return loss
    pos = F.normalize(pos_tokens, dim=-1)  # [B,Kp,C]
    # 正对数似然
    logits_pos = torch.einsum('bc,bkc->bk', q, pos) / temperature  # [B,Kp]
    # 负样本
    if neg_tokens.numel() > 0:
        neg  = F.normalize(neg_tokens, dim=-1)
        logits_neg = torch.einsum('bc,bnc->bn', q, neg) / temperature  # [B,Kn]
        # NCE：每个正样本与所有负样本组成对比
        logsumexp_neg = torch.logsumexp(logits_neg, dim=1, keepdim=True)  # [B,1]
        nce = - (logits_pos - logsumexp_neg).mean()
    else:
        # 只有正样本时，退化成 -mean(sim)
        nce = - logits_pos.mean()
    return nce

def _gather_tokens_by_index(tokens, idx_mat):
    B, N, C = tokens.shape
    if not isinstance(idx_mat, torch.Tensor) or idx_mat.numel() == 0:
        return tokens.new_zeros((B,0,C))
    K = idx_mat.shape[1]
    out = []
    for bi in range(B):
        sel = idx_mat[bi]
        out.append(tokens[bi, sel, :])
    return torch.stack(out, dim=0)  # [B,K,C]

# ===== SG-PLR & DAAD helpers (minimal) =====
import torch.nn.functional as F
from torchvision.ops import batched_nms, box_iou

def _dictlist_append(d, key, value):
    """确保 d[key] 是 list；若是 Tensor，先按 dim=0 拆成 list 再 append。"""
    if key not in d:
        d[key] = []
    else:
        v = d[key]
        # 如果已经是 list，直接用；如果是 Tensor，按样本维拆成 list
        if torch.is_tensor(v):
            # v.shape: [K, ...] -> [Tensor(...)] * K
            d[key] = [t for t in v.unbind(dim=0)]
        elif not isinstance(v, list):
            # 避免遇到 ndarray 或其它类型，做一次“列表化”
            try:
                d[key] = list(v)
            except Exception:
                d[key] = [v]
    d[key].append(value)

def _collect_attn_maps_from(model):
    caches = []
    holders = []  # 记录持有缓存的模块
    for m in model.modules():
        if hasattr(m, "_saliency_cache") and isinstance(m._saliency_cache, list) and len(m._saliency_cache) > 0:
            maps = [x[1] for x in m._saliency_cache if x is not None and isinstance(x, tuple)]
            if len(maps) > 0:
                caches.extend(maps)
                holders.append(m)

    if len(caches) == 0:
        return None

    # 统一尺寸 + 归一
    maxH = max([t.shape[-2] for t in caches])
    maxW = max([t.shape[-1] for t in caches])
    up = []
    for t in caches:
        if t.shape[-2:] != (maxH, maxW):
            t = F.interpolate(t, size=(maxH, maxW), mode='bilinear', align_corners=False)
        t = t / (t.amax(dim=(2,3), keepdim=True) + 1e-6)
        up.append(t)
    attn_mean = torch.stack(up, dim=0).mean(0)  # [B,1,H,W]

    # ★ 清空，防止跨 batch 叠加
    for m in holders:
        m._saliency_cache = []

    return attn_mean


def _boxes_to_gaussian_heatmap(boxes_xyxy, scores, out_hw, img_hw, sigma_ratio=0.15, min_score=0.0):
    """
    将一张图的多个 xyxy 框与分数，投影为特征图尺寸的高斯热图（归一化到[0,1]）。
    - boxes_xyxy: Tensor[K,4] ；坐标既可为像素也可为 [0,1] 归一化坐标（自动判断）
    - scores:     Tensor[K]
    - out_hw:     (Hout, Wout) 目标热图尺寸（例如注意力图尺寸）
    - img_hw:     (Himg, Wimg) 该图原图尺寸（用于归一化坐标→像素坐标转换）
    """
    device = boxes_xyxy.device
    Hout, Wout = out_hw
    Himg, Wimg = img_hw
    # 坐标归一化处理：若坐标最大值<=1.5，视为归一化，需放大到像素坐标
    if boxes_xyxy.numel() > 0 and boxes_xyxy.max() <= 1.5:
        scale = torch.tensor([Wimg, Himg, Wimg, Himg], device=device)
        boxes_xyxy = boxes_xyxy * scale

    yy = torch.linspace(0, Himg-1, Hout, device=device).view(Hout, 1).expand(Hout, Wout)
    xx = torch.linspace(0, Wimg-1, Wout, device=device).view(1, Wout).expand(Hout, Wout)
    heat = torch.zeros((Hout, Wout), device=device)

    for b, s in zip(boxes_xyxy, scores):
        if s < min_score:
            continue
        x1, y1, x2, y2 = b
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = (x2 - x1).clamp(min=1.0)
        bh = (y2 - y1).clamp(min=1.0)
        # sigma 与框尺寸相关，sigma_ratio 可微调（0.10~0.25）
        sigma_x = max(bw.item()*sigma_ratio, 1.0)
        sigma_y = max(bh.item()*sigma_ratio, 1.0)
        g = torch.exp(-0.5 * (((xx - cx)/sigma_x)**2 + ((yy - cy)/sigma_y)**2))
        heat = torch.maximum(heat, s * g)  # 取逐像素最大可避免多目标过度平均

    if heat.max() > 0:
        heat = heat / (heat.max() + 1e-6)
    return heat  # [Hout, Wout]


def _compute_box_saliency(attn_map, box_xyxy, img_hw):
    """
    在注意力图上计算单个框区域的显著性均值。
    - attn_map: [1,Hout,Wout] for single image
    - box_xyxy: Tensor[4], 支持像素或[0,1]坐标
    - img_hw: (Himg,Wimg)
    """
    Hout, Wout = attn_map.shape[-2:]
    Himg, Wimg = img_hw
    b = box_xyxy.clone()
    if b.max() <= 1.5:
        b *= torch.tensor([Wimg, Himg, Wimg, Himg], device=b.device)
    x1, y1, x2, y2 = b
    # 映射到注意力图坐标
    x1 = (x1 / (Wimg-1) * (Wout-1)).clamp(0, Wout-1).long()
    x2 = (x2 / (Wimg-1) * (Wout-1)).clamp(0, Wout-1).long()
    y1 = (y1 / (Himg-1) * (Hout-1)).clamp(0, Hout-1).long()
    y2 = (y2 / (Himg-1) * (Hout-1)).clamp(0, Hout-1).long()
    if (x2 <= x1) or (y2 <= y1):
        return torch.tensor(0.0, device=attn_map.device)
    patch = attn_map[..., y1:y2+1, x1:x2+1]
    return patch.mean().detach()

def _canonicalize_pseudo_keys_to_batch_index(idx_list, labels_dict, boxes_dict, scores_dict, targets_val):
    """
    统一把 idx_list 和 三个字典的 key 都转换为“本批次的样本索引 bi (0..B-1)”。
    如果原本已经是 bi，保持不变；如果是 image_id，则用映射表转换。
    """
    # 建立 映射: image_id -> bi
    id2bi = {targets_val[bi]['image_id'].item(): bi for bi in range(len(targets_val))}

    def _map_key(k):
        # 已是批内索引
        if isinstance(k, int) and 0 <= k < len(targets_val):
            return k
        # 是 image_id
        if k in id2bi:
            return id2bi[k]
        # 其余情况（字符串/张量等）尝试转 int 再判定
        try:
            kk = int(k)
            if 0 <= kk < len(targets_val):
                return kk
            if kk in id2bi:
                return id2bi[kk]
        except Exception:
            pass
        # 找不到映射就返回 None，后续会丢弃该条
        return None

    def _remap_dict(d):
        newd = {}
        for k, v in list(d.items()):
            nk = _map_key(k)
            if nk is None:  # 无法映射，丢弃这项，避免越界
                continue
            newd.setdefault(nk, v)
        return newd

    # 规范 idx_list
    idx_list_new = []
    for x in idx_list:
        nx = _map_key(int(x) if not isinstance(x, int) else x)
        if nx is not None:
            idx_list_new.append(nx)
    # 去重并排序，避免重复索引
    idx_list_new = sorted(set(idx_list_new))

    labels_dict  = _remap_dict(labels_dict)
    boxes_dict   = _remap_dict(boxes_dict)
    scores_dict  = _remap_dict(scores_dict)

    # 最后做一道 sanity check
    if len(idx_list_new) > 0:
        assert max(idx_list_new) < len(targets_val), "idx_list contains indices >= batch size"
    return idx_list_new, labels_dict, boxes_dict, scores_dict

def _tensorize_pseudo_targets(pseudo_targets, device):
    """
    将 pseudo_targets[bi] 的 'boxes'/'labels'/'scores' 从 list 统一转换为 Tensor：
      - boxes:  float32, [K,4]
      - labels: int64,   [K]
      - scores: float32, [K]
    兼容空集：返回形状为 [0,4]/[0]/[0] 的 Tensor
    """
    for k, d in pseudo_targets.items():
        # 1) boxes
        boxes = d.get('boxes', [])
        if isinstance(boxes, list):
            if len(boxes) == 0:
                boxes_t = torch.empty((0, 4), dtype=torch.float32, device=device)
            else:
                # 保证每个元素都是 1D/4D Tensor；若有 numpy/标量，as_tensor 统一
                boxes_t = torch.stack([torch.as_tensor(b, dtype=torch.float32, device=device).view(-1) for b in boxes], dim=0)
        elif torch.is_tensor(boxes):
            boxes_t = boxes.to(device=device, dtype=torch.float32)
        else:
            # 其他类型兜底为空
            boxes_t = torch.empty((0, 4), dtype=torch.float32, device=device)
        d['boxes'] = boxes_t

        # 2) labels
        labels = d.get('labels', [])
        if isinstance(labels, list):
            if len(labels) == 0:
                labels_t = torch.empty((0,), dtype=torch.int64, device=device)
            else:
                labels_t = torch.stack([torch.as_tensor(lb, dtype=torch.int64, device=device).view(()) for lb in labels], dim=0)
        elif torch.is_tensor(labels):
            labels_t = labels.to(device=device, dtype=torch.int64)
        else:
            labels_t = torch.empty((0,), dtype=torch.int64, device=device)
        d['labels'] = labels_t

        # 3) scores
        scores = d.get('scores', [])
        if isinstance(scores, list):
            if len(scores) == 0:
                scores_t = torch.empty((0,), dtype=torch.float32, device=device)
            else:
                scores_t = torch.stack([torch.as_tensor(s, dtype=torch.float32, device=device).view(()) for s in scores], dim=0)
        elif torch.is_tensor(scores):
            scores_t = scores.to(device=device, dtype=torch.float32)
        else:
            scores_t = torch.empty((0,), dtype=torch.float32, device=device)
        d['scores'] = scores_t

        # 可选 sanity check：三者长度一致
        assert d['boxes'].shape[0] == d['labels'].shape[0] == d['scores'].shape[0], \
            f"Inconsistent pseudo sizes at key={k}: boxes={d['boxes'].shape[0]}, labels={d['labels'].shape[0]}, scores={d['scores'].shape[0]}"
    return pseudo_targets

def _resize_samples(samples, scale=0.9):
    """
    仅缩放 samples['tensors']，其余 meta 不动；返回一个“仅用于前向”的浅拷贝。
    你的 Samples 一般是 dict-like：{'tensors': [B,3,H,W], 'mask': ...}
    """
    if isinstance(samples, dict) and 'tensors' in samples:
        x = samples['tensors']
        H, W = x.shape[-2:]
        new_hw = (int(H * scale), int(W * scale))
        xs = F.interpolate(x, size=new_hw, mode='bilinear', align_corners=False)
        s2 = dict(samples)
        s2['tensors'] = xs
        return s2
    return samples

def _stack_if_list(xs, shape4=False, device=None, dtype=None):
    """把 list[Tensor] 统一成 Tensor；支持空集合。shape4=True 时 boxes 输出 [K,4]。"""
    if torch.is_tensor(xs):
        t = xs
        if device is not None: t = t.to(device)
        if dtype  is not None: t = t.to(dtype)
        return t
    if isinstance(xs, list):
        if len(xs) == 0:
            if shape4: return torch.empty((0,4), device=device, dtype=dtype or torch.float32)
            else:      return torch.empty((0,),  device=device, dtype=dtype or torch.float32)
        ts = [torch.as_tensor(z, device=device, dtype=dtype) for z in xs]
        if shape4: ts = [z.view(-1) for z in ts]
        return torch.stack(ts, dim=0)
    # 兜底：空
    if shape4: return torch.empty((0,4), device=device, dtype=dtype or torch.float32)
    else:      return torch.empty((0,),  device=device, dtype=dtype or torch.float32)