import torch
from torch import nn

# =========================
# YAML 读取（按多条路径找，命中即返回）
# =========================
def _ycfg(cfg, key_paths, default=None):
    root = getattr(cfg, "yaml_cfg", None)
    if not isinstance(root, dict):
        return default

    def _get(d, path):
        cur = d
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    for p in key_paths:
        v = _get(root, p)
        if v is not None:
            return v
    return default


# =========================
# 冻结/解冻 与 优化器构建
# =========================
NORM_NAMES = ("norm", "bn", "batchnorm", "layernorm", "groupnorm", "ln", "gn")

def _is_norm_param(name: str, module: nn.Module):
    name_l = name.lower()
    if any(k in name_l for k in NORM_NAMES):
        return True
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)
    for t in norm_types:
        if isinstance(module, t):
            return True
    return False

def freeze_backbone_light_train(model: nn.Module,
                                keep_modules=("encoder.ra_modules", "decoder", "head"),
                                verbose=True):
    total, trainable = 0, 0
    for (name, p) in model.named_parameters():
        total += 1
        keep = False
        # 1) 轻模块/检测头：按前缀保留
        if any(name.startswith(pref) for pref in keep_modules):
            keep = True
        # 2) 各类归一化层参数：总是保留可训
        if _is_norm_param(name, getattr(model, name.split('.')[0], nn.Module())):
            keep = True
        p.requires_grad_(keep)
        if keep: trainable += 1
    if verbose:
        print(f"[freeze] trainable params: {trainable}/{total} (light modules + norms + heads)")

def unfreeze_all(model: nn.Module, verbose=True):
    total = 0
    for _, p in model.named_parameters():
        p.requires_grad_(True); total += 1
    if verbose:
        print(f"[unfreeze] all params trainable: {total}")


# 更新参数的会进入decay和no_decay: decay组，weight_decay=1e-5；no_decay=0
# 1 维参数 / bias / norm → no_decay;其它 → decay
def build_adamw_from_requires_grad(model: nn.Module,
                                   ref_optim: torch.optim.Optimizer) -> torch.optim.AdamW:
    """
    根据已有的 AdamW 优化器 ref_optim 重新构建一个新的 AdamW，
    只保留 requires_grad = True 的参数。
    这样可以保证：
      - param_groups 的划分（哪一组是 backbone、哪一组是 encoder/decoder、norm/no_decay 等）
      - 每组的 lr / weight_decay / betas / eps 等超参数
    都和 YAML 构建出的 optimizer 完全一致。

    注意：这是一个“重建”操作，原来的动量等 state 会重新开始，对 TTA 这种短训练影响很小。
    """

    new_param_groups = []

    for group in ref_optim.param_groups:
        # 只保留当前还需要训练的参数
        new_params = [p for p in group["params"] if p.requires_grad]
        if not new_params:
            # 这一组如果没有任何 trainable 参数，就跳过
            continue

        # 复制除 "params" 之外的所有超参数（lr, weight_decay, betas, eps, 等）
        new_group = {k: v for k, v in group.items() if k != "params"}
        new_group["params"] = new_params
        new_param_groups.append(new_group)

    # 用新的 param_groups 构建 AdamW
    # （类型和 ref_optim 保持一致，这里假定是 AdamW；如果你以后有别的优化器可以再做判断）
    new_optim = torch.optim.AdamW(new_param_groups)
    return new_optim


def build_adamw_from_cfg(model, opt_cfg: dict | None):
    """
    从 optimizer 配置字典构建 AdamW。
    允许 opt_cfg 为空；内部会使用合理默认值。
    """
    if opt_cfg is None:
        opt_cfg = {}
    lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
    return build_adamw_from_requires_grad(
        model, lr=lr, weight_decay=weight_decay, betas=betas
    )
