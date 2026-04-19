"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

# -----------
import numpy as np
from ..self_training.self_training_utils import (get_pseudo_label_via_threshold, deal_pesudo_label,
                                                 rescale_pseudo_targets,
                                                 convert_to_list_format)
import torch.nn.functional as F
from ..method.utils import (_daa_loss_from_feats,_collect_attn_maps_from,_canonicalize_pseudo_keys_to_batch_index,
                            _resize_samples,_dictlist_append,_tensorize_pseudo_targets,_stack_if_list,
                            extract_encoder_tokens_after_attn,sample_salient_tokens,_boxes_to_gaussian_heatmap,
                            _gather_tokens_by_index,info_nce_query_token_loss,_find_adapter_backbone,batched_nms)

from ..RL.rl_adaptation_utils import (
    build_rl_state,
    map_action_to_hparams,
    compute_proxy_reward,
    snapshot_trainable_params,
    measure_param_drift,
    set_train_mode_by_action,
)

# ===== RL helper =====
def restore_requires_grad_from_backup(model, backup_dict):
    if backup_dict is None:
        return
    for n, p in model.named_parameters():
        if n in backup_dict:
            p.requires_grad_(backup_dict[n])

# =========================================================================
def train_one_epoch_eval(model: torch.nn.Module, criterion: torch.nn.Module,
                         data_loader: Iterable, data_loader_val: Iterable,
                         optimizer: torch.optim.Optimizer,
                         device: torch.device, epoch: int, max_norm: float = 0,
                         postprocessor=None, coco_evaluator=None,
                         # ===== RL args =====
                         rl_enable=False, rl_controller=None, rl_optimizer=None,
                         reward_ema=None, rl_entropy_coef=0.01, rl_state_dim=16,
                         rl_greedy_eval=False, **kwargs):

    model.train()
    criterion.train()


    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 1)
    writer: SummaryWriter = kwargs.get('writer', None)

    ema: ModelEMA = kwargs.get('ema', None)
    scaler: GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)


    total_epochs = kwargs.get('total_epochs', None)
    # total_epochs = 10
    print('[total_epochs]:', total_epochs)

    rl_prev_metrics = {
        "ratio_high_conf": 0.0,
        "last_update_action": 1,
        "reward_ma": 0.0,
    }

    base_action_cfg = {
        "hi_score_thr_qt": 0.45,
        "pos_quantile": 0.68,
        "neg_quantile": 0.83,
        "lam_qt_base": 0.022,
        "lambda_attn": 0.10,
        "daa_weight": 0.20,

        # ===== SG-PLR related =====
        "max_rescue_per_img": 6,
        "sgplr_low_min": 0.25,
        "sgplr_low_max": 0.35,
        "sgplr_phi": 0.68,
        "sgplr_hi_thr": 0.40,
    }

    reward_cfg = {
        "w_hq": float(kwargs.get("rl_w_hq", 1.0)),
        "w_attn": float(kwargs.get("rl_w_attn", 0.5)),
        "w_qt": float(kwargs.get("rl_w_qt", 0.5)),
        "w_daa": float(kwargs.get("rl_w_daa", 0.3)),
        "w_drift": float(kwargs.get("rl_w_drift", 0.4)),
    }

    total_epochs_for_rl = total_epochs if total_epochs is not None else 10

    rl_log = {
        "reward": 0.0,
        "policy_loss": 0.0,
        "pseudo_action": 0.0,
        "align_action": 0.0,
        "update_action": 0.0,
    }

    teacher_model = ema.module
    teacher_model.eval()
    alpha_ema = 0.999

    # -------- coco eval config ----------
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    criterion.eval()

    # -------------------------
    # helpers: move postprocess outputs to CPU to save VRAM
    # -------------------------
    def _pp_out_to_cpu(pp_out_list):
        # pp_out_list: list[dict{'boxes','scores','labels',...}] on GPU
        # return same structure but tensors on CPU (non_blocking)
        out = []
        for d in pp_out_list:
            nd = {}
            for k, v in d.items():
                if torch.is_tensor(v):
                    nd[k] = v.detach().to('cpu', non_blocking=True)
                else:
                    nd[k] = v
            out.append(nd)
        return out

    def _pp_out_item_to_device(d, dev):
        # one image dict cpu -> device (only tensors)
        nd = {}
        for k, v in d.items():
            if torch.is_tensor(v):
                nd[k] = v.to(dev, non_blocking=True)
            else:
                nd[k] = v
        return nd

    def _slice_samples(s, a, b):
        # 兼容 dict nested tensor: {'tensors':[B,3,H,W], 'mask':[B,H,W]}
        if isinstance(s, dict) and 'tensors' in s:
            out = dict(s)
            out['tensors'] = s['tensors'][a:b]
            if 'mask' in s and s['mask'] is not None:
                out['mask'] = s['mask'][a:b]
            return out
        if torch.is_tensor(s):
            return s[a:b]
        raise TypeError(f"Unsupported samples type: {type(s)}")

    def _slice_list(xs, a, b):
        return xs[a:b]

    for i, ((samples, targets, samples_val, targets_val), (_, _, _, _)) in enumerate(
            zip(metric_logger.log_every(data_loader, print_freq, header), data_loader_val)):

        samples = samples.to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        samples_val = samples_val.to(device, non_blocking=True)
        targets_val = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets_val]

        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        # ===== reset per-iter RL logs =====
        rl_log = {
            "reward": 0.0,
            "policy_loss": 0.0,
            "pseudo_action": 0.0,
            "align_action": 0.0,
            "update_action": 0.0,
            "sgplr_low_min": 0.0,
            "sgplr_low_max": 0.0,
            "sgplr_phi": 0.0,
            "sgplr_hi_thr": 0.0,
            "sgplr_max_rescue": 0.0,
            "batch_low_cand": 0.0,
            "batch_rescued": 0.0,
        }

        # ==========================================================
        # 1) teacher
        # ==========================================================
        with torch.no_grad():
            teacher_predict_results = teacher_model(samples_val)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets_val], dim=0)
            teacher_predict_results = postprocessor(teacher_predict_results, orig_target_sizes)

        teacher_predict_results_cpu = _pp_out_to_cpu(teacher_predict_results)
        del teacher_predict_results
        del orig_target_sizes
        torch.cuda.empty_cache() if kwargs.get("force_empty_cache", False) else None

        # ==========================================================
        # ==========================================================
        num_classes = 4
        threshold = np.asarray([0.3] * num_classes)

        idx_list, labels_dict, boxes_dict, scores_dict = get_pseudo_label_via_threshold(
            teacher_predict_results_cpu, threshold=threshold
        )

        idx_list, labels_dict, boxes_dict, scores_dict = _canonicalize_pseudo_keys_to_batch_index(
            idx_list, labels_dict, boxes_dict, scores_dict, targets_val
        )
        # ==========================================================
        # unified RL action pre-sampling for SG-PLR + later reward/policy update
        # ==========================================================
        sgplr_action_cfg = {
            "max_rescue_per_img": base_action_cfg["max_rescue_per_img"],
            "sgplr_low_min": base_action_cfg["sgplr_low_min"],
            "sgplr_low_max": base_action_cfg["sgplr_low_max"],
            "sgplr_phi": base_action_cfg["sgplr_phi"],
            "sgplr_hi_thr": base_action_cfg["sgplr_hi_thr"],
        }

        # ==========================================================
        # current teacher-side detection quality statistics for RL state/reward
        # ==========================================================
        high_thr = float(kwargs.get("rl_high_conf_thr", 0.50))
        mid_low_thr = float(kwargs.get("rl_mid_conf_low_thr", 0.30))
        mid_high_thr = float(kwargs.get("rl_mid_conf_high_thr", 0.50))

        total_boxes_all = 0
        total_high_conf = 0
        total_mid_conf = 0
        score_sum_all = 0.0

        B_teacher = len(teacher_predict_results_cpu)
        for res_cpu in teacher_predict_results_cpu:
            if ("scores" not in res_cpu) or (res_cpu["scores"] is None):
                continue
            scores_cpu = res_cpu["scores"]
            if scores_cpu.numel() == 0:
                continue

            total_boxes_all += int(scores_cpu.numel())
            total_high_conf += int((scores_cpu >= high_thr).sum().item())
            total_mid_conf += int(((scores_cpu >= mid_low_thr) & (scores_cpu < mid_high_thr)).sum().item())
            score_sum_all += float(scores_cpu.float().sum().item())

        ratio_high_conf = float(total_high_conf / max(1, total_boxes_all))
        ratio_mid_conf = float(total_mid_conf / max(1, total_boxes_all))
        num_boxes_per_img = float(total_boxes_all / max(1, B_teacher))
        mean_teacher_score = float(score_sum_all / max(1, total_boxes_all))

        rl_action = None
        action_cfg = map_action_to_hparams(
            pseudo_action=1, align_action=1, update_action=2, base_cfg=base_action_cfg
        )

        if rl_enable and rl_controller is not None:
            pre_rl_state_metrics = {
                "ratio_high_conf": ratio_high_conf,
                "ratio_mid_conf": ratio_mid_conf,
                "num_boxes_per_img": num_boxes_per_img,
                "mean_teacher_score": mean_teacher_score,
                "attn_loss": rl_prev_metrics.get("attn_loss", 0.0),
                "qt_loss": rl_prev_metrics.get("qt_loss", 0.0),
                "daa_preserve": rl_prev_metrics.get("daa_preserve", 0.0),
                "daa_un": rl_prev_metrics.get("daa_un", 0.0),
                "token_cov": rl_prev_metrics.get("token_cov", 0.0),
                "fg_bg_gap": rl_prev_metrics.get("fg_bg_gap", 0.0),
                "base_loss": rl_prev_metrics.get("base_loss", 0.0),
                "param_drift": rl_prev_metrics.get("param_drift", 0.0),
                "grad_norm": rl_prev_metrics.get("grad_norm", 0.0),
                "reward_ma": rl_prev_metrics.get("reward_ma", 0.0),
            }

            pre_rl_state = build_rl_state(
                device=device,
                epoch=epoch,
                total_epochs=total_epochs_for_rl,
                prev_metrics=rl_prev_metrics,
                cur_metrics=pre_rl_state_metrics,
                state_dim=rl_state_dim,
            )

            if rl_greedy_eval:
                rl_action = rl_controller.act_greedy(pre_rl_state)
            else:
                rl_action = rl_controller.sample_action(pre_rl_state)

            action_cfg = map_action_to_hparams(
                pseudo_action=rl_action.pseudo_action,
                align_action=rl_action.align_action,
                update_action=rl_action.update_action,
                base_cfg=base_action_cfg,
            )

        sgplr_action_cfg["max_rescue_per_img"] = action_cfg["max_rescue_per_img"]
        sgplr_action_cfg["sgplr_low_min"] = action_cfg["sgplr_low_min"]
        sgplr_action_cfg["sgplr_low_max"] = action_cfg["sgplr_low_max"]
        sgplr_action_cfg["sgplr_phi"] = action_cfg["sgplr_phi"]
        sgplr_action_cfg["sgplr_hi_thr"] = action_cfg["sgplr_hi_thr"]

        # ==========================================================
        # 3) SG-PLR
        # ==========================================================
        model.eval()
        with torch.no_grad():
            _ = model(samples_val)
        attn_val = _collect_attn_maps_from(model)
        del _
        with torch.no_grad():
            samples_val_s = _resize_samples(samples_val, scale=0.9)
            _ = model(samples_val_s)
        attn_val_s = _collect_attn_maps_from(model)
        del _
        del samples_val_s
        model.train()

        if attn_val is not None:
            B = attn_val.shape[0]
            batch_low_cand = 0
            batch_rescued = 0

            low_min = sgplr_action_cfg["sgplr_low_min"]
            low_max = sgplr_action_cfg["sgplr_low_max"]
            sgplr_phi = sgplr_action_cfg["sgplr_phi"]
            sgplr_hi_thr = sgplr_action_cfg["sgplr_hi_thr"]
            sgplr_max_rescue = sgplr_action_cfg["max_rescue_per_img"]

            from torchvision.ops import box_iou

            for bi in range(B):

                res_cpu = teacher_predict_results_cpu[bi]
                if ("scores" not in res_cpu) or (res_cpu["scores"] is None) or (res_cpu["scores"].numel() == 0):
                    continue

                res = _pp_out_item_to_device(res_cpu, device)
                scores = res["scores"]
                boxes = res["boxes"]
                labels = res["labels"]

                h_img, w_img = targets_val[bi]["orig_size"].tolist()

                mask_low = (scores >= low_min) & (scores < low_max)
                num_low = int(mask_low.sum().item())
                if num_low > 0:
                    batch_low_cand += num_low

                if mask_low.any():
                    max_rescue_per_img = sgplr_max_rescue
                    rescued = 0
                    idxs = torch.nonzero(mask_low, as_tuple=False).squeeze(1)
                    order = torch.argsort(scores[idxs], descending=True)

                    for j in order:
                        if rescued >= max_rescue_per_img:
                            break
                        b = boxes[idxs[j]]
                        s = scores[idxs[j]]
                        lb = labels[idxs[j]]

                        Hs, Ws = attn_val.shape[-2:]
                        cx = 0.5 * (b[0] + b[2])
                        cy = 0.5 * (b[1] + b[3])
                        xx = int((cx / (w_img - 1) * (Ws - 1)).clamp(0, Ws - 1))
                        yy = int((cy / (h_img - 1) * (Hs - 1)).clamp(0, Hs - 1))
                        y1 = max(0, yy - 1);
                        y2 = min(Hs - 1, yy + 1)
                        x1 = max(0, xx - 1);
                        x2 = min(Ws - 1, xx + 1)
                        center_sal = attn_val[bi, 0, y1:y2 + 1, x1:x2 + 1].mean()

                        if attn_val_s is None:
                            continue
                        Hs2, Ws2 = attn_val_s.shape[-2:]
                        xx2 = int((cx / (w_img - 1) * (Ws2 - 1)).clamp(0, Ws2 - 1))
                        yy2 = int((cy / (h_img - 1) * (Hs2 - 1)).clamp(0, Hs2 - 1))
                        y1s = max(0, yy2 - 1);
                        y2s = min(Hs2 - 1, yy2 + 1)
                        x1s = max(0, xx2 - 1);
                        x2s = min(Ws2 - 1, xx2 + 1)
                        center_sal_s = attn_val_s[bi, 0, y1s:y2s + 1, x1s:x2s + 1].mean()

                        phi = sgplr_phi
                        if float(center_sal) < phi or float(center_sal_s) < phi:
                            continue

                        hi_mask = scores >= sgplr_hi_thr
                        if hi_mask.any():
                            hi_boxes = boxes[hi_mask]
                            ious = box_iou(b.unsqueeze(0), hi_boxes).squeeze(0)
                            if float(ious.max()) < 0.2:
                                continue

                        if bi not in idx_list:
                            idx_list.append(bi)
                        _dictlist_append(boxes_dict, bi, b.detach().to('cpu'))
                        _dictlist_append(labels_dict, bi, lb.detach().to('cpu'))
                        _dictlist_append(scores_dict, bi, torch.tensor(0.30).cpu())
                        rescued += 1

                    batch_rescued += rescued
                    idx_list = sorted(set(idx_list))
                    # RL logging for SG-PLR rescue policy
                    if rl_enable:
                        rl_log["sgplr_low_min"] = float(low_min)
                        rl_log["sgplr_low_max"] = float(low_max)
                        rl_log["sgplr_phi"] = float(sgplr_phi)
                        rl_log["sgplr_hi_thr"] = float(sgplr_hi_thr)
                        rl_log["sgplr_max_rescue"] = float(sgplr_max_rescue)
                        rl_log["batch_low_cand"] = float(batch_low_cand)
                        rl_log["batch_rescued"] = float(batch_rescued)

                # res (GPU)
                del res, scores, boxes, labels

            for bi in idx_list:
                bxs = _stack_if_list(boxes_dict.get(bi, []), shape4=True, device='cpu', dtype=torch.float32)
                lbs = _stack_if_list(labels_dict.get(bi, []), shape4=False, device='cpu', dtype=torch.int64)
                scs = _stack_if_list(scores_dict.get(bi, []), shape4=False, device='cpu', dtype=torch.float32)
                if bxs.numel() == 0:
                    continue
                keep = batched_nms(bxs, scs, lbs, iou_threshold=0.5)
                bxs = bxs[keep];
                lbs = lbs[keep];
                scs = scs[keep]
                boxes_dict[bi] = [t for t in bxs]
                labels_dict[bi] = [t for t in lbs]
                scores_dict[bi] = [t for t in scs]

        # deal_pesudo_label
        target_pseudo_labels = deal_pesudo_label(targets_val, idx_list, labels_dict, boxes_dict, scores_dict)
        target_pseudo_labels = _tensorize_pseudo_targets(target_pseudo_labels, device=device)
        target_pseudo_labels = rescale_pseudo_targets(samples_val, target_pseudo_labels)
        target_pseudo_labels_list = convert_to_list_format(target_pseudo_labels)
        # ---- ensure pseudo list is batch-aligned (length == B) to avoid empty slice ----
        B = samples['tensors'].shape[0] if isinstance(samples, dict) else len(targets_val)

        def _empty_pseudo_like(t):
            dev = t["image_id"].device
            return {
                "boxes": torch.empty((0, 4), dtype=torch.float32, device=dev),
                "labels": torch.empty((0,), dtype=torch.int64, device=dev),
                "scores": torch.empty((0,), dtype=torch.float32, device=dev),
                "image_id": t["image_id"],
                "orig_size": t.get("orig_size", None),
                "size": t.get("size", None),
            }

        # case1: list is compacted by idx_list (common in self-training code)
        if len(target_pseudo_labels_list) != B:
            full = [_empty_pseudo_like(targets_val[i]) for i in range(B)]

            if len(target_pseudo_labels_list) == len(idx_list):
                for bi, pd in zip(idx_list, target_pseudo_labels_list):
                    # 防御：补齐缺失 key
                    if "labels" not in pd: pd["labels"] = torch.empty((0,), dtype=torch.int64, device=device)
                    if "boxes" not in pd: pd["boxes"] = torch.empty((0, 4), dtype=torch.float32, device=device)
                    if "scores" not in pd: pd["scores"] = torch.empty((0,), dtype=torch.float32, device=device)
                    if "image_id" not in pd: pd["image_id"] = targets_val[bi]["image_id"]
                    full[bi] = pd
            else:
                mp = {}
                for pd in target_pseudo_labels_list:
                    if "image_id" in pd:
                        mp[int(pd["image_id"]) if torch.is_tensor(pd["image_id"]) else int(pd["image_id"])] = pd
                for i in range(B):
                    iid = int(targets_val[i]["image_id"])
                    if iid in mp:
                        full[i] = mp[iid]

            target_pseudo_labels_list = full

        # ==========================================================
        # 4) TTA ：student forward + loss
        # ==========================================================
        if len(idx_list) >= 1:
            # =========================================================
            # Micro-batch accumulate
            # =========================================================
            loss_dict_sum = None
            loss_dict = {}

            micro = int(kwargs.get("microbatch", 2))  # 推荐 4；仍 OOM 就设为 2
            B = samples['tensors'].shape[0] if isinstance(samples, dict) else len(targets)
            assert B % micro == 0, f"Batch {B} must be divisible by microbatch {micro}"
            num_chunks = B // micro

            # ---- DAA static config ----
            daa_beta_alpha = float(kwargs.get('daa_beta_alpha', 1.0))
            daa_noise_std = float(kwargs.get('daa_noise_std', 0.01))
            daa_temp = float(kwargs.get('daa_temperature', 0.2))
            daa_w_preserve = float(kwargs.get('daa_w_preserve', 1.0))
            daa_w_un = float(kwargs.get('daa_w_un', 1.0))

            # ---- DAAD static config ----
            T_attn = 1.0
            start_epoch_qt = 4
            dist_rel_thr = 0.016

            # =========================================================
            # RL decision block: decide current-step adaptation policy
            # =========================================================
            cur_state_metrics = {
                "ratio_high_conf": ratio_high_conf,
                "ratio_mid_conf": ratio_mid_conf,
                "num_boxes_per_img": num_boxes_per_img,
                "mean_teacher_score": mean_teacher_score,
                "attn_loss": rl_prev_metrics.get("attn_loss", 0.0),
                "qt_loss": rl_prev_metrics.get("qt_loss", 0.0),
                "daa_preserve": rl_prev_metrics.get("daa_preserve", 0.0),
                "daa_un": rl_prev_metrics.get("daa_un", 0.0),
                "token_cov": rl_prev_metrics.get("token_cov", 0.0),
                "fg_bg_gap": rl_prev_metrics.get("fg_bg_gap", 0.0),
                "base_loss": rl_prev_metrics.get("base_loss", 0.0),
                "param_drift": rl_prev_metrics.get("param_drift", 0.0),
                "grad_norm": rl_prev_metrics.get("grad_norm", 0.0),
                "reward_ma": rl_prev_metrics.get("reward_ma", 0.0),
            }

            if rl_enable and rl_controller is not None:
                if rl_action is None:
                    state = build_rl_state(
                        device=device,
                        epoch=epoch,
                        total_epochs=total_epochs_for_rl,
                        prev_metrics=rl_prev_metrics,
                        cur_metrics=cur_state_metrics,
                        state_dim=rl_state_dim,
                    )

                    if rl_greedy_eval:
                        rl_action = rl_controller.act_greedy(state)
                    else:
                        rl_action = rl_controller.sample_action(state)

                    action_cfg = map_action_to_hparams(
                        pseudo_action=rl_action.pseudo_action,
                        align_action=rl_action.align_action,
                        update_action=rl_action.update_action,
                        base_cfg=base_action_cfg,
                    )
            else:
                rl_action = None
                action_cfg = map_action_to_hparams(
                    pseudo_action=1, align_action=1, update_action=2, base_cfg=base_action_cfg
                )

            hi_score_thr_qt = action_cfg["hi_score_thr_qt"]
            pos_quantile = action_cfg["pos_quantile"]
            neg_quantile = action_cfg["neg_quantile"]
            lam_qt_base = action_cfg["lam_qt_base"]
            lambda_attn_cur = action_cfg["lambda_attn"]
            daa_weight_cur = action_cfg["daa_weight"]
            max_rescue_per_img = action_cfg["max_rescue_per_img"]

            skip_update = action_cfg["skip_update"]
            adapter_only = action_cfg["adapter_only"]
            full_update = action_cfg["full_update"]

            # =========================================================
            # =========================================================
            def _slice_samples(s, a, b):
                if isinstance(s, dict) and 'tensors' in s:
                    out = {'tensors': s['tensors'][a:b]}
                    if 'mask' in s and s['mask'] is not None:
                        out['mask'] = s['mask'][a:b]
                    else:
                        out['mask'] = None
                    return out
                if torch.is_tensor(s):
                    return s[a:b]
                raise TypeError(f"Unsupported samples type: {type(s)}")

            # =========================================================
            # =========================================================
            optimizer.zero_grad(set_to_none=True)

            # RL: snapshot params before current step, for drift reward
            ref_params = snapshot_trainable_params(model) if rl_enable else None

            # RL: backup requires_grad before adapter-only mode
            reqgrad_backup = None
            if rl_enable and adapter_only:
                reqgrad_backup = {n: p.requires_grad for n, p in model.named_parameters()}
                set_train_mode_by_action(model, adapter_only=True)

            total_loss_accum = 0.0

            # =========================================================
            # L_qt: batch-level cov accumulator (no grad)
            # =========================================================
            pos_sum_all = 0.0
            N_all = 0.0

            # =========================================================
            # RL reward/state
            # =========================================================
            attn_loss_sum = 0.0
            qt_loss_sum = 0.0
            daa_preserve_sum = 0.0
            daa_un_sum = 0.0
            fg_bg_gap_sum = 0.0
            fg_bg_gap_count = 0

            for ck in range(num_chunks):
                a = ck * micro
                b = (ck + 1) * micro

                attn_loss_mb_scalar = 0.0
                qt_loss_mb_scalar = 0.0
                daa_preserve_mb_scalar = 0.0
                daa_un_mb_scalar = 0.0
                fg_bg_gap_mb_scalar = 0.0

                # -----------------------------
                # micro-batch
                # -----------------------------
                samples_mb = _slice_samples(samples, a, b)
                targets_mb = targets[a:b]

                samples_val_mb = _slice_samples(samples_val, a, b)
                targets_val_mb = targets_val[a:b]

                pseudo_mb = target_pseudo_labels_list[a:b]
                teacher_cpu_mb = teacher_predict_results_cpu[a:b]  # list[dict] on CPU

                # =========================================================
                # 1) Student forward
                # =========================================================
                outputs = model(samples_mb)

                # =========================================================
                # 2) q_vec
                # =========================================================
                q_embed = outputs['aux_queries'] if 'aux_queries' in outputs else outputs.get('hs', None)
                if q_embed is None:
                    raise KeyError("outputs must contain 'aux_queries' or 'hs' for q_vec construction")

                if q_embed.dim() == 4:
                    q_embed_last = q_embed[-1]  # [B,Q,C]
                else:
                    q_embed_last = q_embed  # [B,Q,C]

                q_prob = outputs['pred_logits'].softmax(-1)  # [B,Q,num_cls]
                conf, _ = q_prob.max(-1)  # [B,Q]
                Qm = 12
                topq = torch.topk(conf, k=min(Qm, conf.shape[1]), dim=1).indices  # [B,Qm]
                q_vec = torch.gather(
                    q_embed_last, 1,
                    topq.unsqueeze(-1).expand(-1, -1, q_embed_last.size(-1))
                ).mean(1)  # [B,C]

                # =========================================================
                # 3) det loss（严格用 pseudo_mb，不混用全 batch）
                # =========================================================
                loss_dict = criterion(outputs, pseudo_mb, **metas)
                base_loss = sum(loss_dict.values())
                # --- 累加成 batch 级 loss_dict（不改算法，只用于日志）---
                if loss_dict_sum is None:
                    loss_dict_sum = {k: v.detach() for k, v in loss_dict.items()}
                else:
                    for k, v in loss_dict.items():
                        loss_dict_sum[k] = loss_dict_sum[k] + v.detach()

                # =========================================================
                # 4) L_qt（token-level，teacher_cpu_mb + targets_val_mb）
                # =========================================================
                add_lqt = torch.zeros((), device=device)
                tokens, Hs, Ws = extract_encoder_tokens_after_attn(model)[:3]

                if tokens is not None:
                    Bmb, N, C = tokens.shape  # Bmb == micro
                    sal = tokens.norm(p=2, dim=-1)
                    sal = sal / (sal.amax(dim=1, keepdim=True) + 1e-6)

                    mask_pos = torch.zeros((Bmb, N), dtype=torch.bool, device=device)
                    mask_neg = torch.zeros((Bmb, N), dtype=torch.bool, device=device)

                    for bi_local in range(Bmb):
                        # teacher 结果：来自 teacher_cpu_mb
                        res = _pp_out_item_to_device(teacher_cpu_mb[bi_local], device)
                        boxes = res["boxes"]
                        scores = res["scores"]

                        h_img, w_img = targets_val_mb[bi_local]["orig_size"].tolist()

                        hi = scores >= hi_score_thr_qt
                        boxes_hi = boxes[hi]
                        if boxes_hi.numel() == 0:
                            continue

                        phi_pos = torch.quantile(sal[bi_local], pos_quantile).item()
                        phi_neg = torch.quantile(sal[bi_local], neg_quantile).item()

                        ctrs = 0.5 * (boxes_hi[:, :2] + boxes_hi[:, 2:])
                        diag = float((w_img ** 2 + h_img ** 2) ** 0.5)

                        # pos
                        for n_idx in range(N):
                            yy = n_idx // Ws
                            xx = n_idx % Ws
                            if sal[bi_local, n_idx] < phi_pos:
                                continue
                            cx = (xx / (Ws - 1)) * (w_img - 1)
                            cy = (yy / (Hs - 1)) * (h_img - 1)
                            inside = ((cx >= boxes_hi[:, 0]) & (cy >= boxes_hi[:, 1]) &
                                      (cx <= boxes_hi[:, 2]) & (cy <= boxes_hi[:, 3])).any().item()
                            if inside:
                                mask_pos[bi_local, n_idx] = True
                                continue
                            dmin = float(torch.sqrt(((ctrs[:, 0] - cx) ** 2 + (ctrs[:, 1] - cy) ** 2).min()))
                            if dmin / (diag + 1e-6) < dist_rel_thr:
                                mask_pos[bi_local, n_idx] = True

                        # neg
                        for n_idx in range(N):
                            if mask_pos[bi_local, n_idx]:
                                continue
                            if sal[bi_local, n_idx] < phi_neg:
                                continue
                            yy = n_idx // Ws
                            xx = n_idx % Ws
                            cx = (xx / (Ws - 1)) * (w_img - 1)
                            cy = (yy / (Hs - 1)) * (h_img - 1)
                            dmin = float(torch.sqrt(((ctrs[:, 0] - cx) ** 2 + (ctrs[:, 1] - cy) ** 2).min()))
                            if dmin / (diag + 1e-6) > 0.03:
                                mask_neg[bi_local, n_idx] = True

                        del res, boxes, scores, boxes_hi, ctrs

                    pos_per_img = mask_pos.view(Bmb, -1).sum(dim=1).float()
                    # accumulate batch-level coverage stats (detach)
                    pos_sum_all += float(pos_per_img.sum().detach().item())
                    N_all += float(Bmb * N)

                    cov = (pos_per_img / float(N)).mean() if pos_per_img.numel() > 0 else torch.tensor(0.0,
                                                                                                       device=device)
                    cov = torch.clamp(cov, min=0.02, max=0.08)
                    scale_cov = torch.sqrt(cov)

                    K_pos_use = 64
                    pos_idx, _ = sample_salient_tokens(tokens, sal, mask_pos, mask_pos, k_pos=K_pos_use, k_neg=0)
                    neg_cap = max(16, K_pos_use // 2)
                    _, neg_idx = sample_salient_tokens(tokens, sal, mask_neg, mask_neg, k_pos=0, k_neg=neg_cap)

                    pos_tok = _gather_tokens_by_index(tokens, pos_idx)
                    neg_tok = _gather_tokens_by_index(tokens, neg_idx)

                    L_qt = info_nce_query_token_loss(q_vec, pos_tok, neg_tok, temperature=0.09)
                    if epoch >= start_epoch_qt:
                        phase = min(1.0, float(epoch - start_epoch_qt) / max(1, (total_epochs - start_epoch_qt)))
                        add_lqt = (lam_qt_base * phase * scale_cov) * L_qt

                    del tokens, sal, mask_pos, mask_neg, pos_tok, neg_tok, pos_idx, neg_idx, pos_per_img, cov, scale_cov, L_qt

                # =========================================================
                # 5) DAAD
                # =========================================================
                L_attn_full = torch.zeros((), device=device)

                # 1) teacher on micro-batch, not full batch
                with torch.inference_mode():
                    t_out_train_mb = teacher_model(samples_mb)
                    orig_sizes_train_mb = torch.stack([t["orig_size"] for t in targets_mb], dim=0)
                    t_out_train_mb = postprocessor(t_out_train_mb, orig_sizes_train_mb)
                del orig_sizes_train_mb

                # 2) student on micro-batch only for attention cache
                model.eval()
                with torch.no_grad():
                    _ = model(samples_mb)
                attn_train_mb = _collect_attn_maps_from(model)
                del _
                model.train()

                if (attn_train_mb is not None) and (len(t_out_train_mb) == attn_train_mb.shape[0]) and (
                        lambda_attn_cur > 0):
                    Bfull, _, Hm, Wm = attn_train_mb.shape
                    for bi_full in range(Bfull):
                        res = t_out_train_mb[bi_full]
                        if ("boxes" not in res) or (res["boxes"].numel() == 0):
                            continue
                        boxes = res["boxes"]
                        scores = res["scores"]
                        h_img, w_img = targets_mb[bi_full]["orig_size"].tolist()

                        heat = _boxes_to_gaussian_heatmap(
                            boxes, scores, (Hm, Wm), (h_img, w_img),
                            sigma_ratio=0.12, min_score=0.30
                        )

                        att = attn_train_mb[bi_full, 0]
                        att_t = (att / T_attn).clamp(min=1e-6)
                        att_t = att_t / (att_t.max() + 1e-6)

                        m = 0.5 * (att_t + heat)
                        js = 0.5 * F.kl_div((att_t + 1e-6).log(), (m + 1e-6), reduction='batchmean') \
                             + 0.5 * F.kl_div((heat + 1e-6).log(), (m + 1e-6), reduction='batchmean')
                        L_attn_full = L_attn_full + js

                del t_out_train_mb, attn_train_mb

                # =========================================================
                # 6) DAA
                # =========================================================
                daa_preserve = torch.zeros((), device=device)
                daa_un = torch.zeros((), device=device)

                backbone = _find_adapter_backbone(model)
                if backbone is not None and hasattr(backbone, 'adapters_mod0') and hasattr(backbone, 'enable_adapters'):
                    prev_flag = bool(backbone.enable_adapters)

                    # r(x): no adapter, no grad
                    backbone.enable_adapters = False
                    with torch.no_grad():
                        feats_no = backbone(samples_val_mb)

                    # DAA(r(x)): with adapter, need grad (only adapters trainable)
                    backbone.enable_adapters = True
                    feats_ad = backbone(samples_val_mb)

                    # restore
                    backbone.enable_adapters = prev_flag

                    daa_preserve, daa_un = _daa_loss_from_feats(
                        backbone=backbone,
                        feats_no=feats_no,
                        feats_ad=feats_ad,
                        beta_alpha=daa_beta_alpha,
                        noise_std=daa_noise_std,
                        temperature=daa_temp
                    )

                    # free asap
                    del feats_no, feats_ad

                daa_loss = daa_weight_cur * (daa_w_preserve * daa_preserve + daa_w_un * daa_un)

                # =========================================================
                # 7) total loss
                # =========================================================
                daad_term = (lambda_attn_cur * L_attn_full) / num_chunks
                total_loss_mb = base_loss + daad_term + add_lqt + daa_loss

                if not skip_update:
                    (total_loss_mb / num_chunks).backward()

                total_loss_accum += float(total_loss_mb.detach().item())
                last_attn_loss_scalar = (
                    float(L_attn_full.detach().item())
                    if ("L_attn_full" in locals() and L_attn_full is not None)
                    else 0.0
                )

                last_qt_loss_scalar = (
                    float(L_qt.detach().item())
                    if ("L_qt" in locals() and L_qt is not None)
                    else 0.0
                )

                last_daa_preserve_scalar = (
                    float(daa_preserve.detach().item())
                    if ("daa_preserve" in locals() and daa_preserve is not None)
                    else 0.0
                )

                last_daa_un_scalar = (
                    float(daa_un.detach().item())
                    if ("daa_un" in locals() and daa_un is not None)
                    else 0.0
                )

                # =========================================================
                # 8) micro-batch
                # =========================================================
                del outputs, base_loss
                del q_embed, q_embed_last, q_prob, conf, topq, q_vec
                del daa_preserve, daa_un, daa_loss, total_loss_mb
                del pseudo_mb, teacher_cpu_mb, targets_mb, targets_val_mb, samples_mb, samples_val_mb

            # 将累计的 loss_dict 变成 batch=8 的“平均”loss_dict，供 reduce_dict/log 使用
            if loss_dict_sum is not None:
                loss_dict = {k: (v / num_chunks) for k, v in loss_dict_sum.items()}
            else:
                loss_dict = {}

            # =========================================================
            # =========================================================
            if not skip_update:
                if max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                else:
                    grad_norm = torch.tensor(0.0, device=device)

                optimizer.step()
            else:
                grad_norm = torch.tensor(0.0, device=device)
                optimizer.zero_grad(set_to_none=True)

            # restore requires_grad after adapter-only update
            if rl_enable and reqgrad_backup is not None:
                restore_requires_grad_from_backup(model, reqgrad_backup)

        else:
            loss_dict = {}
            base_loss = torch.tensor(0.0, device=device)

        # ==========================================================
        # RL: reward + policy update
        # ==========================================================
        if len(idx_list) >= 1:

            cur_metrics = {
                "ratio_high_conf": float(locals().get("ratio_high_conf", 0.0)),
                "ratio_mid_conf": float(locals().get("ratio_mid_conf", 0.0)),
                "num_boxes_per_img": float(locals().get("num_boxes_per_img", 0.0)),
                "mean_teacher_score": float(locals().get("mean_teacher_score", 0.0)),
                "attn_loss": float(locals().get("last_attn_loss_scalar", 0.0)),
                "qt_loss": float(locals().get("last_qt_loss_scalar", 0.0)),
                "daa_preserve": float(locals().get("last_daa_preserve_scalar", 0.0)),
                "daa_un": float(locals().get("last_daa_un_scalar", 0.0)),
                "token_cov": float(locals().get("scale_cov", 0.0)),
                "fg_bg_gap": float(locals().get("fg_bg_gap", 0.0)),
                "base_loss": float(locals().get("total_loss_accum", 0.0)),
                "grad_norm": float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            }

            if rl_enable and ref_params is not None and not skip_update:
                cur_metrics["param_drift"] = measure_param_drift(model, ref_params)
            else:
                cur_metrics["param_drift"] = 0.0

            if rl_enable and rl_controller is not None and rl_optimizer is not None and rl_action is not None:
                reward_val = compute_proxy_reward(
                    cur_metrics=cur_metrics,
                    prev_metrics=rl_prev_metrics,
                    reward_cfg=reward_cfg,
                )

                reward_ma_val = reward_ema.update(reward_val) if reward_ema is not None else reward_val
                cur_metrics["reward_ma"] = reward_ma_val
                cur_metrics["last_update_action"] = rl_action.update_action

                advantage = reward_val - reward_ma_val

                policy_loss = -rl_action.log_prob * torch.tensor(
                    advantage, dtype=torch.float32, device=device
                ) - rl_entropy_coef * rl_action.entropy.mean()

                rl_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                rl_optimizer.step()

                rl_log["reward"] += float(reward_val)
                rl_log["policy_loss"] += float(policy_loss.detach().item())
                rl_log["pseudo_action"] += float(rl_action.pseudo_action)
                rl_log["align_action"] += float(rl_action.align_action)
                rl_log["update_action"] += float(rl_action.update_action)
            else:
                cur_metrics["reward_ma"] = rl_prev_metrics.get("reward_ma", 0.0)
                cur_metrics["last_update_action"] = 1

            rl_prev_metrics = cur_metrics

        # ==========================================================
        # 5) inference + coco evaluator
        # ==========================================================
        model.eval()
        criterion.eval()
        with torch.no_grad():
            outputs_eval = teacher_model(samples_val)

        if len(idx_list) >= 1:
            loss_dict_eval = criterion(outputs_eval, target_pseudo_labels_list, **metas)
        else:
            loss_dict_eval = criterion(outputs_eval, targets_val, **metas)

        orig_target_sizes_eval = torch.stack([t["orig_size"] for t in targets_val], dim=0)
        results = postprocessor(outputs_eval, orig_target_sizes_eval)

        res = {target['image_id'].item(): output for target, output in zip(targets_val, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)


        del outputs_eval, results, res, orig_target_sizes_eval, loss_dict_eval

        model.train()

        # ==========================================================
        # 6) EMA update
        # ==========================================================
        with torch.no_grad():
            student_model_state_dict = model.state_dict()
            teacher_model_state_dict = teacher_model.state_dict()
            for entry in teacher_model_state_dict.keys():
                teacher_param = teacher_model_state_dict[entry].clone().detach()
                student_param = student_model_state_dict[entry].clone().detach()
                new_param = (teacher_param * alpha_ema) + (student_param * (1. - alpha_ema))
                teacher_model_state_dict[entry] = new_param
            teacher_model.load_state_dict(teacher_model_state_dict)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        # ==========================================================
        # 7) logging
        # ==========================================================
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values()) if len(loss_dict_reduced) > 0 else torch.tensor(0.0)

        if not math.isfinite(float(loss_value)):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if rl_enable:
            metric_logger.update(
                rl_reward=rl_log["reward"],
                rl_policy_loss=rl_log["policy_loss"],
                rl_pseudo_action=rl_log["pseudo_action"],
                rl_align_action=rl_log["align_action"],
                rl_update_action=rl_log["update_action"],
            )

            if writer is not None and dist_utils.is_main_process():
                global_step = epoch * len(data_loader) + i
                writer.add_scalar("RL/sgplr_low_min", rl_log.get("sgplr_low_min", 0.0), global_step)
                writer.add_scalar("RL/sgplr_low_max", rl_log.get("sgplr_low_max", 0.0), global_step)
                writer.add_scalar("RL/sgplr_phi", rl_log.get("sgplr_phi", 0.0), global_step)
                writer.add_scalar("RL/sgplr_hi_thr", rl_log.get("sgplr_hi_thr", 0.0), global_step)
                writer.add_scalar("RL/sgplr_max_rescue", rl_log.get("sgplr_max_rescue", 0.0), global_step)
                writer.add_scalar("RL/batch_low_cand", rl_log.get("batch_low_cand", 0.0), global_step)
                writer.add_scalar("RL/batch_rescued", rl_log.get("batch_rescued", 0.0), global_step)

        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', float(loss_value), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', float(v), global_step)

        # 释放本 iter 常驻 CPU teacher 结果（避免 CPU 内存涨）
        del teacher_predict_results_cpu
        del idx_list, labels_dict, boxes_dict, scores_dict, target_pseudo_labels, target_pseudo_labels_list
        del attn_val, attn_val_s

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader,
             coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    for samples, targets, samples1, targets1 in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.tensor([[samples.shape[-1], samples.shape[-2]]], device=samples.device)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator



