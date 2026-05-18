import argparse
import csv
import os
import random
import time
import math
from datetime import timedelta
import json
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from dataset import FullDataset
from model.model_last import DMNetWithClustering

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ==================== DDP 工具函数 ====================

def setup_ddp():
    """初始化 DDP 进程组，返回 local_rank。"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ==================== 原有工具函数（不变） ====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_int_csv(text: str) -> Tuple[int, ...]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Cannot parse int values from: {text}")
    return tuple(values)


def str2bool(v):
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


ABLATION_MODES = {
    "manual": None,
    "full": {
        "use_consistency_correction": True,
        "use_boundary_propagation": True,
        "use_prototype_attention_pooling": True,
        "use_cluster_entropy": True,
        "use_interval_grade": True,
        "use_dino_cond_refine": True,
        "roi_use_anomaly": True,
        "use_leaf_proxy": True,
    },
    "no_consistency": {"use_consistency_correction": False},
    "no_boundary": {"use_boundary_propagation": False},
    "no_proto_pool": {"use_prototype_attention_pooling": False},
    "no_cluster_entropy": {"use_cluster_entropy": False},
    "no_proto_diversity": {"lambda_proto_diversity": 0.0},
    "no_interval": {"use_interval_grade": False},
    "baseline": {
        "use_consistency_correction": False,
        "use_boundary_propagation": False,
        "use_prototype_attention_pooling": False,
        "use_cluster_entropy": False,
        "use_interval_grade": False,
        "use_dino_cond_refine": False,
        "roi_use_anomaly": False,
        "use_leaf_proxy": False,
    },
}


def apply_ablation_mode(args):
    mode = str(getattr(args, "ablation_mode", "full")).strip().lower()
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation_mode={mode}, expected one of {list(ABLATION_MODES.keys())}")
    cfg = ABLATION_MODES[mode]
    if cfg is not None:
        if mode == "full":
            for key, value in cfg.items():
                setattr(args, key, value)
        else:
            full = ABLATION_MODES["full"] or {}
            merged = dict(full)
            merged.update(cfg)
            for key, value in merged.items():
                setattr(args, key, value)
    return args


def resolve_grade_columns(csv_path, image_col_hint, label_col_hint):
    if not csv_path or not os.path.exists(csv_path):
        return None, None
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [x.strip() for x in (reader.fieldnames or [])]
    if not fields:
        raise RuntimeError(f"CSV has no header: {csv_path}")
    image_candidates = [image_col_hint, "image", "image_name", "filename", "file", "name"]
    label_candidates = [label_col_hint, "class", "grade", "label", "grade_label"]
    image_col = next((c for c in image_candidates if c and c in fields), None)
    label_col = next((c for c in label_candidates if c and c in fields), None)
    if image_col is None:
        raise KeyError(f"Cannot resolve image column in {csv_path}. header={fields}")
    if label_col is None:
        raise KeyError(f"Cannot resolve grade/class column in {csv_path}. header={fields}")
    return image_col, label_col


def _stem(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(str(path_or_name).strip()))[0]


def load_grade_mapping(csv_path, image_col, label_col):
    mapping: Dict[str, int] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = _stem(row[image_col])
            mapping[stem] = int(float(row[label_col]))
    return mapping


def compute_grade_class_weights(csv_path, image_col, label_col, grade_values, power=0.5, clip=3.0):
    if not csv_path or not image_col or not label_col or not os.path.exists(csv_path):
        return None
    counter: Counter = Counter()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gv = int(float(row[label_col]))
            counter[gv] += 1
    if not counter:
        return None
    counts = np.array([float(counter.get(int(g), 0)) for g in grade_values], dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    inv = (counts.max() / counts) ** float(power)
    inv = inv / (inv.mean() + 1e-12)
    if clip > 1.0:
        inv = np.clip(inv, 1.0 / float(clip), float(clip))
    return torch.tensor(inv, dtype=torch.float32)


class LegacyGradeWrapper(Dataset):
    def __init__(self, base_dataset, grade_map, grade_values):
        self.base = base_dataset
        self.grade_map = grade_map
        self.grade_values = tuple(int(v) for v in grade_values)
        self.grade_to_idx = {v: i for i, v in enumerate(self.grade_values)}

    def __len__(self):
        return len(self.base)

    def _sample_stem(self, idx):
        if hasattr(self.base, "ids"):
            ids = getattr(self.base, "ids")
            if isinstance(ids, (list, tuple)) and idx < len(ids):
                return _stem(ids[idx])
        if hasattr(self.base, "images"):
            images = getattr(self.base, "images")
            if isinstance(images, (list, tuple)) and idx < len(images):
                return _stem(images[idx])
        raise RuntimeError("Legacy dataset has no usable ids/images attribute for grade alignment.")

    def __getitem__(self, idx):
        sample = self.base[idx]
        if not isinstance(sample, dict):
            raise RuntimeError("Expected dataset sample dict with keys 'image' and 'label'.")
        stem = self._sample_stem(idx)
        if stem not in self.grade_map:
            raise KeyError(f"Missing grade label for sample: {stem}")
        grade_value = int(self.grade_map[stem])
        if grade_value not in self.grade_to_idx:
            raise ValueError(f"Grade value {grade_value} not in grade_values={self.grade_values}")
        sample["grade_idx"] = torch.tensor(self.grade_to_idx[grade_value], dtype=torch.long)
        sample["grade_value"] = torch.tensor(float(grade_value), dtype=torch.float32)
        sample["grade_norm"] = torch.tensor(float(grade_value) / float(self.grade_values[-1]), dtype=torch.float32)
        sample["image_id"] = stem
        return sample


def build_dataset_compat(image_root, mask_root, size, mode, grade_csv, grade_image_col, grade_label_col, grade_values, mean, std):
    try:
        ds = FullDataset(
            image_root, mask_root, size, mode=mode, grade_csv=grade_csv,
            grade_image_col=grade_image_col or "image", grade_label_col=grade_label_col or "grade",
            grade_values=grade_values, mean=mean, std=std,
        )
        return ds, bool(grade_csv)
    except TypeError:
        ds = FullDataset(image_root, mask_root, size, mode)
        if grade_csv and grade_image_col and grade_label_col:
            grade_map = load_grade_mapping(grade_csv, grade_image_col, grade_label_col)
            ds = LegacyGradeWrapper(ds, grade_map, grade_values)
            return ds, True
        return ds, False


def default_grade_bounds(grade_values, delta):
    if tuple(grade_values) == (0, 1, 3, 5, 7, 9):
        return [(0.0, delta), (delta, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50), (0.50, 1.0)]
    n = len(grade_values)
    edges = np.linspace(0.0, 1.0, num=n + 1).tolist()
    return [(edges[i], edges[i + 1]) for i in range(n)]


def parse_grade_bounds(spec, expected_levels):
    raw_ranges = [chunk.strip() for chunk in spec.split(";") if chunk.strip()]
    bounds = []
    for chunk in raw_ranges:
        lo, hi = map(float, chunk.split(","))
        bounds.append((lo, hi))
    if len(bounds) != expected_levels:
        raise ValueError(f"grade_bounds count={len(bounds)} but expected_levels={expected_levels}")
    return bounds


def format_seconds(seconds):
    return str(timedelta(seconds=int(seconds)))


def linear_ramp(epoch, start_epoch, end_epoch):
    if epoch < start_epoch: return 0.0
    if epoch >= end_epoch or end_epoch <= start_epoch: return 1.0
    return float(epoch - start_epoch) / float(end_epoch - start_epoch)


def ratio_to_grade_idx(ratio, bounds):
    idx = torch.full_like(ratio, len(bounds) - 1, dtype=torch.long)
    for i, (lo, hi) in enumerate(bounds):
        mask = (ratio >= lo) & (ratio < hi) if i < len(bounds) - 1 else (ratio >= lo) & (ratio <= hi)
        idx[mask] = i
    return idx


def ordinal_targets(grade_idx, num_levels):
    thresholds = torch.arange(num_levels - 1, device=grade_idx.device).unsqueeze(0)
    return (grade_idx.unsqueeze(1) > thresholds).float()


def ordinal_loss(grade_logits, grade_idx, num_levels, sample_weight=None):
    targets = ordinal_targets(grade_idx, num_levels)
    bce = F.binary_cross_entropy_with_logits(grade_logits, targets, reduction="none").mean(dim=1)
    if sample_weight is not None:
        sw = sample_weight.float().clamp_min(1e-6)
        return (bce * sw).sum() / sw.sum()
    return bce.mean()


def ordinal_monotonicity_loss(grade_logits):
    p_gt = torch.sigmoid(grade_logits)
    if p_gt.shape[1] <= 1: return grade_logits.new_tensor(0.0)
    violation = p_gt[:, 1:] - p_gt[:, :-1]
    return F.relu(violation).mean()


def weighted_seg_loss(logits, target, pixel_weight, class_weight):
    ce = F.cross_entropy(logits, target, weight=class_weight, reduction="none")
    ce = (ce * pixel_weight).sum() / (pixel_weight.sum() + 1e-6)
    prob = F.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    w = pixel_weight.unsqueeze(1)
    inter = (prob * one_hot * w).sum(dim=(2, 3))
    denom = ((prob + one_hot) * w).sum(dim=(2, 3))
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    dice_loss = ((1.0 - dice) * class_weight.view(1, -1)).sum(dim=1) / class_weight.sum()
    return ce + dice_loss.mean()


def inside_leaf_loss(leaf_total_prob, disease_prob):
    numer = (disease_prob * (1.0 - leaf_total_prob)).sum(dim=(1, 2))
    denom = disease_prob.sum(dim=(1, 2)) + 1e-6
    return (numer / denom).mean()


def interval_grade_loss(disease_ratio, grade_idx, bounds, beta, sample_weight=None):
    bound_tensor = torch.tensor(bounds, device=disease_ratio.device, dtype=disease_ratio.dtype)
    lower, upper = bound_tensor[grade_idx, 0], bound_tensor[grade_idx, 1]
    loss = F.softplus(beta * (lower - disease_ratio)) + F.softplus(beta * (disease_ratio - upper))
    if sample_weight is not None:
        sw = sample_weight.float().clamp_min(1e-6)
        return (loss * sw).sum() / sw.sum()
    return loss.mean()


def boundary_target_from_mask(mask):
    edge = torch.zeros_like(mask, dtype=torch.bool)
    edge[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    edge[:, :-1, :] |= mask[:, :-1, :] != mask[:, 1:, :]
    edge[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    edge[:, :, :-1] |= mask[:, :, :-1] != mask[:, :, 1:]
    return edge.unsqueeze(1).float()


def boundary_supervision_loss(boundary_logits, target_mask, pos_weight=4.0):
    if not boundary_logits:
        return target_mask.new_tensor(0.0, dtype=torch.float32)
    target = boundary_target_from_mask(target_mask).float()
    loss = target.new_tensor(0.0)
    pw = target.new_tensor(float(pos_weight))
    for logit in boundary_logits:
        pred = F.interpolate(logit, size=target.shape[2:], mode="bilinear", align_corners=False)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pw)
        prob = torch.sigmoid(pred)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0))
        loss = loss + bce + dice.mean()
    return loss / float(len(boundary_logits))


def consistency_correction_loss(aux, sample_weight=None):
    if "grade_score_head" not in aux or "ratio_grade_score" not in aux:
        return torch.tensor(0.0, device=next(iter(aux.values())).device)
    score_gap = F.smooth_l1_loss(aux["grade_score_head"], aux["ratio_grade_score"], reduction="none")
    inconsistency = aux.get("inconsistency_score", score_gap.detach())
    if sample_weight is not None:
        sw = sample_weight.float().clamp_min(1e-6)
        score_term = (score_gap * sw).sum() / sw.sum()
        incons_term = (inconsistency * sw).sum() / sw.sum()
    else:
        score_term = score_gap.mean()
        incons_term = inconsistency.mean()
    if "grade_probs_head" in aux and "ratio_grade_probs" in aux:
        head_log = (aux["grade_probs_head"].clamp_min(1e-8)).log()
        ratio_prob = aux["ratio_grade_probs"].clamp_min(1e-8)
        kl = F.kl_div(head_log, ratio_prob, reduction="batchmean")
    else:
        kl = score_term.new_tensor(0.0)
    return score_term + 0.5 * kl + 0.25 * incons_term


def cluster_entropy_regularization_loss(aux, target_mask, disease_class_ids, source="gt", bg_weight=1.0):
    if "cluster_probs" not in aux:
        z = target_mask.new_tensor(0.0, dtype=torch.float32)
        return z, z
    cluster_probs = aux["cluster_probs"]
    bsz, n_token, k = cluster_probs.shape
    side = int(n_token ** 0.5)
    if side * side != n_token:
        z = target_mask.new_tensor(0.0, dtype=torch.float32)
        return z, z
    log_k = math.log(float(max(k, 2)))
    entropy = -(cluster_probs.clamp_min(1e-8) * cluster_probs.clamp_min(1e-8).log()).sum(dim=-1) / log_k
    if str(source).lower() == "pred":
        disease_prob = aux.get("disease_prob_coarse", aux["disease_prob"]).detach()
        disease_grid = F.interpolate(disease_prob.unsqueeze(1), size=(side, side), mode="bilinear", align_corners=False).squeeze(1)
    else:
        disease_mask = torch.zeros_like(target_mask, dtype=torch.bool)
        for cid in disease_class_ids:
            disease_mask |= target_mask == int(cid)
        disease_grid = F.interpolate(disease_mask.float().unsqueeze(1), size=(side, side), mode="nearest").squeeze(1)
    disease_w = disease_grid.reshape(bsz, -1).clamp(0.0, 1.0)
    bg_w = (1.0 - disease_w).clamp(0.0, 1.0)
    if aux.get("cluster_token_weight") is not None:
        token_w = aux["cluster_token_weight"].detach().reshape(bsz, -1).clamp(0.0, 1.0)
        disease_w = disease_w * token_w
        bg_w = bg_w * token_w
    low_entropy = (entropy * disease_w).sum() / disease_w.sum().clamp_min(1e-6)
    high_entropy = ((1.0 - entropy) * bg_w).sum() / bg_w.sum().clamp_min(1e-6)
    entropy_loss = low_entropy + float(bg_weight) * high_entropy
    weighted_probs = cluster_probs
    if aux.get("cluster_token_weight") is not None:
        tw = aux["cluster_token_weight"].detach().reshape(bsz, n_token, 1).clamp(0.0, 1.0)
        weighted_probs = weighted_probs * tw
    usage = weighted_probs.sum(dim=(0, 1))
    usage = usage / usage.sum().clamp_min(1e-8)
    div_loss = (usage * (usage.clamp_min(1e-8).log() - math.log(1.0 / float(k)))).sum()
    return entropy_loss, div_loss


def macro_f1(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return float(np.mean(f1s))


def quadratic_weighted_kappa(y_true, y_pred, num_classes):
    conf = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1.0
    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(conf.sum(), 1.0)
    w = np.zeros_like(conf)
    denom = float((num_classes - 1) ** 2) + 1e-8
    for i in range(num_classes):
        for j in range(num_classes):
            w[i, j] = ((i - j) ** 2) / denom
    num = (w * conf).sum()
    den = (w * expected).sum() + 1e-8
    return float(1.0 - num / den)


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def save_confusion_matrix(cm, labels, save_path, title):
    if plt is None: return
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def evaluate(model, loader, device, num_classes, num_grade_levels, grade_bounds, return_details=False):
    # DDP 下传入 model.module 或直接传 model 均可，这里统一处理
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    pred_area = np.zeros(num_classes, dtype=np.float64)
    gt_area = np.zeros(num_classes, dtype=np.float64)
    grade_true, grade_pred_ratio, grade_pred_head, grade_pred_raw, grade_pred_last = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device).long()
            seg_logits, aux = raw_model(x, return_aux=True)
            seg_pred = torch.argmax(seg_logits, dim=1)
            for c in range(num_classes):
                pred_c = seg_pred == c
                gt_c = y == c
                inter[c] += (pred_c & gt_c).sum().item()
                union[c] += (pred_c | gt_c).sum().item()
                pred_area[c] += pred_c.sum().item()
                gt_area[c] += gt_c.sum().item()
            if "grade_idx" in batch:
                gt_idx = batch["grade_idx"].to(device).long()
                ratio_idx = ratio_to_grade_idx(aux["disease_ratio"], grade_bounds)
                head_idx = torch.argmax(aux["grade_probs"], dim=1)
                raw_idx = torch.argmax(aux.get("grade_probs_head", aux["grade_probs"]), dim=1)
                grade_true.extend(gt_idx.cpu().numpy().tolist())
                grade_pred_ratio.extend(ratio_idx.cpu().numpy().tolist())
                grade_pred_head.extend(head_idx.cpu().numpy().tolist())
                grade_pred_raw.extend(raw_idx.cpu().numpy().tolist())
                if "grade_probs_last" in aux:
                    last_idx = torch.argmax(aux["grade_probs_last"], dim=1)
                    grade_pred_last.extend(last_idx.cpu().numpy().tolist())

    tp = inter
    fp = pred_area - tp
    fn = gt_area - tp
    total = gt_area.sum() + 1e-8
    tn = total - tp - fp - fn
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fscore = 2.0 * precision * recall / (precision + recall + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    acc = (tp + tn) / (total + 1e-8)

    metrics = {
        "mIoU": float(np.mean(iou)), "mAcc": float(np.mean(acc)), "mDice": float(np.mean(dice)),
        "mFscore": float(np.mean(fscore)), "mPrecision": float(np.mean(precision)),
        "mRecall": float(np.mean(recall)),
    }
    for c in range(num_classes):
        metrics[f"IoU_{c}"] = float(iou[c])
        metrics[f"Acc_{c}"] = float(acc[c])
        metrics[f"Dice_{c}"] = float(dice[c])
        metrics[f"Fscore_{c}"] = float(fscore[c])
        metrics[f"Precision_{c}"] = float(precision[c])
        metrics[f"Recall_{c}"] = float(recall[c])

    if grade_true:
        gt = np.array(grade_true, dtype=np.int64)
        pred_ratio = np.array(grade_pred_ratio, dtype=np.int64)
        pred_head = np.array(grade_pred_head, dtype=np.int64)
        pred_raw = np.array(grade_pred_raw, dtype=np.int64)
        metrics.update({
            "grade_acc_ratio": float((gt == pred_ratio).mean()),
            "grade_f1_ratio": macro_f1(gt, pred_ratio, num_grade_levels),
            "grade_qwk_ratio": quadratic_weighted_kappa(gt, pred_ratio, num_grade_levels),
            "grade_acc_head": float((gt == pred_head).mean()),
            "grade_f1_head": macro_f1(gt, pred_head, num_grade_levels),
            "grade_qwk_head": quadratic_weighted_kappa(gt, pred_head, num_grade_levels),
            "grade_acc_head_raw": float((gt == pred_raw).mean()),
            "grade_f1_head_raw": macro_f1(gt, pred_raw, num_grade_levels),
            "grade_qwk_head_raw": quadratic_weighted_kappa(gt, pred_raw, num_grade_levels),
        })
        if len(grade_pred_last) == len(grade_true):
            pred_last = np.array(grade_pred_last, dtype=np.int64)
            metrics.update({
                "grade_acc_last": float((gt == pred_last).mean()),
                "grade_f1_last": macro_f1(gt, pred_last, num_grade_levels),
                "grade_qwk_last": quadratic_weighted_kappa(gt, pred_last, num_grade_levels),
            })

    if return_details:
        details = {
            "grade_true": np.asarray(grade_true, dtype=np.int64),
            "grade_pred_ratio": np.asarray(grade_pred_ratio, dtype=np.int64),
            "grade_pred_head": np.asarray(grade_pred_head, dtype=np.int64),
            "grade_pred_head_raw": np.asarray(grade_pred_raw, dtype=np.int64),
            "grade_pred_last": np.asarray(grade_pred_last, dtype=np.int64),
        }
        raw_model.train()
        return metrics, details

    raw_model.train()
    return metrics


def infer_dataset_image_names(dataset):
    if hasattr(dataset, "images") and isinstance(getattr(dataset, "images"), (list, tuple)):
        return [os.path.basename(str(x)) for x in dataset.images]
    if hasattr(dataset, "ids") and isinstance(getattr(dataset, "ids"), (list, tuple)):
        return [str(x) for x in dataset.ids]
    if hasattr(dataset, "base") and isinstance(dataset.base, Dataset):
        return infer_dataset_image_names(dataset.base)
    return []


@torch.no_grad()
def export_prediction_csv(model, loader, device, csv_path, leaf_class_id, disease_class_ids, grade_values, grade_bounds):
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fallback_names = infer_dataset_image_names(loader.dataset)
    cursor = 0
    rows = []
    for batch in loader:
        x = batch["image"].to(device)
        seg_logits, _ = raw_model(x, return_aux=True)
        seg_pred = torch.argmax(seg_logits, dim=1)
        disease_mask = torch.zeros_like(seg_pred, dtype=torch.bool)
        for cid in disease_class_ids:
            disease_mask |= seg_pred == int(cid)
        leaf_mask = (seg_pred == int(leaf_class_id)) | disease_mask
        disease_pixels = disease_mask.sum(dim=(1, 2)).float()
        leaf_pixels = leaf_mask.sum(dim=(1, 2)).float()
        ratio = disease_pixels / (leaf_pixels + 1e-6)
        class_idx = ratio_to_grade_idx(ratio, grade_bounds)
        bsz = int(x.shape[0])
        if "image_id" in batch:
            image_names = [str(v) for v in batch["image_id"]]
        else:
            end = cursor + bsz
            image_names = fallback_names[cursor:end] if end <= len(fallback_names) else [f"sample_{cursor + i:06d}" for i in range(bsz)]
            cursor = end
        for i in range(bsz):
            rows.append({
                "image": image_names[i],
                "leaf_pixels": int(leaf_pixels[i].item()),
                "disease_pixels": int(disease_pixels[i].item()),
                "ratio_percent": float((ratio[i] * 100).item()),
                "class": int(grade_values[int(class_idx[i].item())]),
            })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "leaf_pixels", "disease_pixels", "ratio_percent", "class"])
        writer.writeheader()
        writer.writerows(rows)
    raw_model.train()
    print(f"[Final] prediction list saved: {csv_path} ({len(rows)} rows)")


def init_epoch_metrics_csv(csv_path):
    fields = ["ablation_mode", "epoch", "train_loss", "lr", "skipped_steps", "amp_enabled",
              "downgrade_events", "mIoU", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall",
              "grade_acc_ratio", "grade_f1_ratio", "grade_qwk_ratio",
              "grade_acc_head", "grade_f1_head", "grade_qwk_head",
              "grade_acc_head_raw", "grade_f1_head_raw", "grade_qwk_head_raw",
              "grade_acc_last", "grade_f1_last", "grade_qwk_last"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def append_epoch_metrics_csv(csv_path, ablation_mode, epoch, train_loss, lr, skipped_steps, amp_enabled, downgrade_events, metrics):
    row = {
        "ablation_mode": str(ablation_mode), "epoch": int(epoch), "train_loss": float(train_loss),
        "lr": float(lr), "skipped_steps": int(skipped_steps), "amp_enabled": int(bool(amp_enabled)),
        "downgrade_events": int(downgrade_events),
    }
    for k in ["mIoU", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall",
              "grade_acc_ratio", "grade_f1_ratio", "grade_qwk_ratio",
              "grade_acc_head", "grade_f1_head", "grade_qwk_head",
              "grade_acc_head_raw", "grade_f1_head_raw", "grade_qwk_head_raw",
              "grade_acc_last", "grade_f1_last", "grade_qwk_last"]:
        row[k] = float(metrics.get(k, float("nan")))
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)


def save_final_validation_artifacts(metrics, details, grade_values, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "val_metrics_final.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if not details or details.get("grade_true") is None or details["grade_true"].size == 0: return
    gt = details["grade_true"]
    labels = [str(v) for v in grade_values]
    for key, title in [("grade_pred_ratio", "Ratio Threshold"), ("grade_pred_head", "Head Prediction"),
                       ("grade_pred_head_raw", "Raw Head"), ("grade_pred_last", "Last-Layer Head")]:
        pred = details.get(key)
        if pred is not None and pred.size == gt.size:
            cm = confusion_matrix_np(gt, pred, len(grade_values))
            name = key.split('_')[-1]
            np.savetxt(os.path.join(save_dir, f"confusion_{name}.csv"), cm, fmt="%d", delimiter=",")
            save_confusion_matrix(cm, labels, os.path.join(save_dir, f"confusion_{name}.png"), f"Grade Confusion Matrix ({title})")
    print(f"[Final] confusion matrices saved under: {save_dir}")


def build_argparser():
    p = argparse.ArgumentParser("Train Soybean DM segmentation + disease index model (DDP)")
    p.add_argument("--sam3_path", type=str, required=True)
    p.add_argument("--dinov3_path", type=str, default=None)
    p.add_argument("--dinov3_local_path", type=str, default="./dinov3")
    p.add_argument("--dinov3_model_name", type=str, default="dinov3_vitl16")
    p.add_argument("--dino_lora_rank", type=int, default=0)
    p.add_argument("--dino_lora_alpha", type=float, default=8.0)
    p.add_argument("--dino_lora_dropout", type=float, default=0.0)
    p.add_argument("--dino_intermediate_layer", type=int, default=17)
    p.add_argument("--compare_last_layer_head", type=str2bool, default=True)
    p.add_argument("--train_image_path", type=str, required=True)
    p.add_argument("--train_mask_path", type=str, required=True)
    p.add_argument("--val_image_path", type=str, required=True)
    p.add_argument("--val_mask_path", type=str, required=True)
    p.add_argument("--train_grade_csv", type=str, default="train.csv")
    p.add_argument("--val_grade_csv", type=str, default="val.csv")
    p.add_argument("--grade_image_col", type=str, default="image")
    p.add_argument("--grade_label_col", type=str, default="grade")
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--low_res_size", type=int, default=448)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--leaf_class_id", type=int, default=1)
    p.add_argument("--disease_class_ids", type=str, default="2")
    p.add_argument("--grade_values", type=str, default="0,1,3,5,7,9")
    p.add_argument("--grade_delta", type=float, default=0.003)
    p.add_argument("--grade_bounds", type=str, default="")
    p.add_argument("--beta_start", type=float, default=10.0)
    p.add_argument("--beta_end", type=float, default=35.0)
    p.add_argument("--lambda_grade", type=float, default=0.4)
    p.add_argument("--lambda_inside", type=float, default=0.1)
    p.add_argument("--lambda_ord", type=float, default=1.0)
    p.add_argument("--lambda_intv", type=float, default=1.0)
    p.add_argument("--lambda_cons", type=float, default=0.2)
    p.add_argument("--lambda_mono", type=float, default=0.2)
    p.add_argument("--lambda_grade_last", type=float, default=1.0)
    p.add_argument("--lambda_boundary", type=float, default=0.25)
    p.add_argument("--lambda_cluster_entropy", type=float, default=0.08)
    p.add_argument("--lambda_cluster_diversity", type=float, default=0.02)
    p.add_argument("--lambda_proto_diversity", type=float, default=0.02)
    p.add_argument("--lambda_leaf_proxy", type=float, default=0.1)
    p.add_argument("--boundary_pos_weight", type=float, default=4.0)
    p.add_argument("--use_grade", type=str2bool, default=True)
    p.add_argument("--use_inside", type=str2bool, default=True)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--head_start_epoch", type=int, default=8)
    p.add_argument("--grade_start_epoch", type=int, default=20)
    p.add_argument("--ramp_end_epoch", type=int, default=90)
    p.add_argument("--use_clustering", type=str2bool, default=True)
    p.add_argument("--cluster_num_prototypes", type=int, default=16)
    p.add_argument("--cluster_temperature", type=float, default=0.1)
    p.add_argument("--cluster_token_source", type=str, default="mid", choices=["mid", "last", "blend"])
    p.add_argument("--cluster_roi_source", type=str, default="anomaly_then_leaf",
                   choices=["none", "anomaly", "leaf", "anomaly_then_leaf"])
    p.add_argument("--use_consistency_correction", type=str2bool, default=True)
    p.add_argument("--use_cross_task_attention", type=str2bool, default=True)
    p.add_argument("--use_boundary_propagation", type=str2bool, default=True)
    p.add_argument("--use_soft_roi_cluster", type=str2bool, default=True)
    p.add_argument("--use_prototype_attention_pooling", type=str2bool, default=True)
    p.add_argument("--use_cluster_entropy", type=str2bool, default=True)
    p.add_argument("--use_interval_grade", type=str2bool, default=True)
    p.add_argument("--use_leaf_proxy", type=str2bool, default=True)
    p.add_argument("--grade_use_disease_weighted_pool", type=str2bool, default=True)
    p.add_argument("--use_dino_cond_refine", type=str2bool, default=True)
    p.add_argument("--consistency_temperature", type=float, default=0.3)
    p.add_argument("--cluster_entropy_source", type=str, default="gt", choices=["gt", "pred"])
    p.add_argument("--cluster_entropy_bg_weight", type=float, default=1.0)
    p.add_argument("--roi_use_anomaly", type=str2bool, default=True)
    p.add_argument("--roi_anomaly_power", type=float, default=1.0)
    p.add_argument("--detach_shared_for_last_head", type=str2bool, default=True)
    p.add_argument("--min_backbone_load_ratio", type=float, default=0.0)
    p.add_argument("--ablation_mode", type=str, default="manual", choices=list(ABLATION_MODES.keys()))
    p.add_argument("--use_grade_reweight", type=str2bool, default=True)
    p.add_argument("--grade_reweight_power", type=float, default=0.7)
    p.add_argument("--grade_class_weight", type=float, nargs=6, default=None)
    p.add_argument("--grade_reweight_clip", type=float, default=3.0)
    p.add_argument("--score_ratio_weight", type=float, default=0.15)
    p.add_argument("--score_head_weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--auto_downgrade", type=str2bool, default=True)
    p.add_argument("--nan_patience", type=int, default=3)
    p.add_argument("--lr_drop_factor", type=float, default=0.5)
    p.add_argument("--max_downgrades", type=int, default=4)
    p.add_argument("--norm_mean", type=float, nargs=3, default=[0.620, 0.639, 0.594])
    p.add_argument("--norm_std", type=float, nargs=3, default=[0.245, 0.219, 0.281])
    p.add_argument("--pred_csv_name", type=str, default="prediction_image_list.csv")
    return p


# ==================== 主训练函数 ====================

def main():
    args = build_argparser().parse_args()
    apply_ablation_mode(args)

    # ── DDP 初始化 ──────────────────────────────────────────────
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    main_proc = is_main_process()          # 只有 rank-0 做 IO
    # ────────────────────────────────────────────────────────────

    # 固定随机种子（各进程使用不同种子以保证数据多样性）
    set_seed(args.seed + dist.get_rank())

    if args.ablation_mode not in {"manual", "full"}:
        args.save_path = os.path.join(args.save_path, args.ablation_mode)
    if main_proc:
        os.makedirs(args.save_path, exist_ok=True)
    ddp_barrier()   # 等 rank-0 建完目录再继续

    grade_values = parse_int_csv(args.grade_values)
    disease_class_ids = parse_int_csv(args.disease_class_ids)

    if args.grade_bounds.strip():
        grade_bounds = parse_grade_bounds(args.grade_bounds, expected_levels=len(grade_values))
    else:
        grade_bounds = default_grade_bounds(grade_values, delta=args.grade_delta)

    train_grade_csv = args.train_grade_csv if args.train_grade_csv and os.path.exists(args.train_grade_csv) else None
    val_grade_csv = args.val_grade_csv if args.val_grade_csv and os.path.exists(args.val_grade_csv) else None
    train_image_col, train_label_col = resolve_grade_columns(train_grade_csv, args.grade_image_col, args.grade_label_col)
    val_image_col, val_label_col = resolve_grade_columns(val_grade_csv, args.grade_image_col, args.grade_label_col)

    train_dataset, train_has_grade = build_dataset_compat(
        image_root=args.train_image_path, mask_root=args.train_mask_path, size=args.img_size,
        mode="train", grade_csv=train_grade_csv, grade_image_col=train_image_col,
        grade_label_col=train_label_col, grade_values=grade_values, mean=args.norm_mean, std=args.norm_std,
    )
    val_dataset, val_has_grade = build_dataset_compat(
        image_root=args.val_image_path, mask_root=args.val_mask_path, size=args.img_size,
        mode="val", grade_csv=val_grade_csv, grade_image_col=val_image_col,
        grade_label_col=val_label_col, grade_values=grade_values, mean=args.norm_mean, std=args.norm_std,
    )

    # ── DistributedSampler 替换 shuffle=True ────────────────────
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        drop_last=True,         # 保证每卡 batch 大小一致
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,     # 每张卡的 batch_size；总等效 = batch_size × 2
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # val_loader 只在 rank-0 使用，无需 DistributedSampler
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
    )
    # ────────────────────────────────────────────────────────────

    # ── 构建模型并迁移到对应 GPU ─────────────────────────────────
    model = DMNetWithClustering(
        sam3_checkpoint_path=args.sam3_path, dinov3_weight_path=args.dinov3_path,
        dinov3_local_path=args.dinov3_local_path, dinov3_model_name=args.dinov3_model_name,
        img_size=args.img_size, low_res_size=args.low_res_size, num_classes=args.num_classes,
        grade_values=grade_values, leaf_class_id=args.leaf_class_id, disease_class_ids=disease_class_ids,
        num_prototypes=args.cluster_num_prototypes, use_clustering=args.use_clustering,
        cluster_temperature=args.cluster_temperature, cluster_token_source=args.cluster_token_source,
        cluster_roi_source=args.cluster_roi_source, use_consistency_correction=args.use_consistency_correction,
        use_cross_task_attention=args.use_cross_task_attention,
        use_boundary_propagation=args.use_boundary_propagation,
        use_soft_roi_cluster=args.use_soft_roi_cluster,
        use_prototype_attention_pooling=args.use_prototype_attention_pooling,
        grade_use_disease_weighted_pool=args.grade_use_disease_weighted_pool,
        use_dino_cond_refine=args.use_dino_cond_refine, consistency_temperature=args.consistency_temperature,
        roi_use_anomaly=args.roi_use_anomaly, roi_anomaly_power=args.roi_anomaly_power,
        use_leaf_proxy_head=args.use_leaf_proxy, dino_intermediate_layer=args.dino_intermediate_layer,
        compare_last_layer_head=args.compare_last_layer_head,
        detach_shared_for_last_head=args.detach_shared_for_last_head,
        dino_lora_rank=args.dino_lora_rank, dino_lora_alpha=args.dino_lora_alpha,
        dino_lora_dropout=args.dino_lora_dropout, min_backbone_load_ratio=args.min_backbone_load_ratio,
    ).to(device)

    # ── DDP 包装 ─────────────────────────────────────────────────
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,    # 模型有条件分支，必须开启
    )
    # 访问原始模型用 model.module
    # ────────────────────────────────────────────────────────────

    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim == 1 or name.endswith(".bias"): no_decay_params.append(p)
        else: decay_params.append(p)

    optimizer = AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    class_weight = torch.ones(args.num_classes, device=device)
    for cid in disease_class_ids:
        if 0 <= int(cid) < args.num_classes:
            class_weight[int(cid)] = 5.0

    has_grade_train = bool(args.use_grade and train_has_grade)
    has_grade_val = bool(val_has_grade)
    head_start = max(0, int(args.head_start_epoch))
    grade_start = max(args.grade_start_epoch, args.warmup_epochs)

    grade_class_weight = None
    if has_grade_train:
        if args.grade_class_weight is not None:
            grade_class_weight = torch.tensor(args.grade_class_weight, dtype=torch.float32).to(device)
            if main_proc:
                print(f"[Grade Reweight] (Manual) class weights={grade_class_weight.cpu().numpy().round(4).tolist()}")
        elif args.use_grade_reweight:
            grade_class_weight = compute_grade_class_weights(
                csv_path=train_grade_csv, image_col=train_image_col, label_col=train_label_col,
                grade_values=grade_values, power=args.grade_reweight_power, clip=args.grade_reweight_clip,
            )
            if grade_class_weight is not None:
                grade_class_weight = grade_class_weight.to(device)
                if main_proc:
                    print(f"[Grade Reweight] (Auto) class weights={grade_class_weight.cpu().numpy().round(4).tolist()}")

    best_score, best_epoch = -1e9, -1
    start_time = time.time()
    total_bad_steps, consecutive_nonfinite, downgrade_events = 0, 0, 0

    epoch_metrics_csv = os.path.join(args.save_path, "epoch_metrics.csv")
    if main_proc:
        init_epoch_metrics_csv(epoch_metrics_csv)

    if main_proc:
        print("=" * 90)
        print(f"DDP训练: world_size={dist.get_world_size()}, "
              f"每卡batch={args.batch_size}, 等效总batch={args.batch_size * dist.get_world_size()}, "
              f"accumulation={args.accumulation_steps}")
        print(f"epochs={args.epochs}, lr={args.lr}, AMP={amp_enabled}")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable / 1e6:.2f}M")
        print(f"Grade supervision(train/val)={has_grade_train}/{has_grade_val}")
        print("=" * 90)

    # ==================== 训练循环 ====================
    for epoch in range(args.epochs):
        model.train()
        # ── 关键：每个 epoch 告诉 sampler 当前 epoch，保证不同 epoch 的 shuffle 结果不同
        train_sampler.set_epoch(epoch)

        t0 = time.time()
        losses = []
        bad_steps = 0
        consecutive_nonfinite = 0

        grade_ramp = linear_ramp(epoch, head_start, args.ramp_end_epoch)
        beta = args.beta_start + (args.beta_end - args.beta_start) * grade_ramp
        lambda_grade = args.lambda_grade * grade_ramp if args.use_grade else 0.0

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True).long()
            skip_step = False

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                seg_logits, aux = model(x, return_aux=True)
                pixel_weight = torch.ones_like(y, dtype=torch.float32)
                seg_loss = weighted_seg_loss(seg_logits.float(), y, pixel_weight, class_weight.float())
                in_loss = inside_leaf_loss(aux["leaf_total_prob"].float(), aux["disease_prob"].float()) if args.use_inside else seg_logits.new_tensor(0.0)
                boundary_loss_term = cluster_entropy_loss_term = cluster_div_loss_term = seg_logits.new_tensor(0.0)
                proto_div_loss = leaf_proxy_loss = seg_logits.new_tensor(0.0)

                if args.use_boundary_propagation and "boundary_logits" in aux:
                    b_terms = list(aux["boundary_logits"]) + ([aux["boundary_fused"]] if "boundary_fused" in aux else [])
                    boundary_loss_term = boundary_supervision_loss(b_terms, y, pos_weight=args.boundary_pos_weight)

                if args.use_clustering and args.use_cluster_entropy and "cluster_probs" in aux:
                    cluster_entropy_loss_term, cluster_div_loss_term = cluster_entropy_regularization_loss(
                        aux, y, disease_class_ids, source=args.cluster_entropy_source,
                        bg_weight=args.cluster_entropy_bg_weight)

                if args.use_clustering and args.lambda_proto_diversity > 0:
                    raw_m = model.module
                    if raw_m.clustering is not None:
                        proto_div_loss = aux.get("proto_diversity_loss", raw_m.clustering.prototype_diversity_loss())

                if args.use_leaf_proxy and "leaf_proxy_logits" in aux:
                    bsz_l, grid_l = y.shape[0], int(aux["dino_grid_size"])
                    leaf_logits = aux["leaf_proxy_logits"].view(bsz_l, grid_l, grid_l)
                    disease_mask = torch.zeros_like(y, dtype=torch.bool)
                    for cid in disease_class_ids: disease_mask |= y == cid
                    leaf_mask_grid = F.interpolate(
                        ((y == int(args.leaf_class_id)) | disease_mask).float().unsqueeze(1),
                        size=(grid_l, grid_l), mode="nearest").squeeze(1)
                    leaf_proxy_loss = F.binary_cross_entropy_with_logits(leaf_logits, leaf_mask_grid)

                grade_loss = seg_logits.new_tensor(0.0)
                if has_grade_train and "grade_idx" in batch:
                    grade_idx = batch["grade_idx"].to(device).long()
                    sample_weight = grade_class_weight[grade_idx] if grade_class_weight is not None else None
                    if epoch >= head_start:
                        ord_loss = ordinal_loss(aux["grade_logits"], grade_idx, len(grade_values), sample_weight)
                        mono_loss = ordinal_monotonicity_loss(aux["grade_logits"])
                        grade_loss = grade_loss + args.lambda_ord * ord_loss + args.lambda_mono * mono_loss
                        if "grade_logits_last" in aux and float(args.lambda_grade_last) > 0.0:
                            ord_last = ordinal_loss(aux["grade_logits_last"], grade_idx, len(grade_values), sample_weight)
                            mono_last = ordinal_monotonicity_loss(aux["grade_logits_last"])
                            grade_loss = grade_loss + float(args.lambda_grade_last) * (args.lambda_ord * ord_last + args.lambda_mono * mono_last)
                    if args.use_interval_grade and epoch >= grade_start:
                        intv_loss = interval_grade_loss(aux["disease_ratio"], grade_idx, grade_bounds, beta, sample_weight)
                        if args.use_consistency_correction:
                            cons_loss = consistency_correction_loss(aux, sample_weight)
                        else:
                            sw = sample_weight.float().clamp_min(1e-6) if sample_weight is not None else None
                            diff = F.smooth_l1_loss(aux["grade_score"], aux["disease_ratio"], reduction="none")
                            cons_loss = (diff * sw).sum() / sw.sum() if sw is not None else diff.mean()
                        grade_loss = grade_loss + args.lambda_intv * intv_loss + args.lambda_cons * cons_loss

                total_loss = (
                    seg_loss + (args.lambda_inside if args.use_inside else 0.0) * in_loss
                    + lambda_grade * grade_loss
                    + (args.lambda_boundary if args.use_boundary_propagation else 0.0) * boundary_loss_term
                    + float(args.lambda_cluster_entropy) * cluster_entropy_loss_term
                    + float(args.lambda_cluster_diversity) * cluster_div_loss_term
                    + float(args.lambda_proto_diversity) * proto_div_loss
                    + float(args.lambda_leaf_proxy) * leaf_proxy_loss
                )
                loss_to_backprop = total_loss / args.accumulation_steps

                if not torch.isfinite(total_loss):
                    skip_step = True

            if skip_step:
                bad_steps += 1
                total_bad_steps += 1
                consecutive_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                if (args.auto_downgrade and consecutive_nonfinite >= max(1, int(args.nan_patience))
                        and downgrade_events < int(args.max_downgrades)):
                    if amp_enabled:
                        amp_enabled = False
                        scaler = torch.cuda.amp.GradScaler(enabled=False)
                    for i, pg in enumerate(optimizer.param_groups):
                        new_lr = max(float(pg["lr"]) * float(args.lr_drop_factor), float(args.min_lr))
                        pg["lr"] = new_lr
                        if hasattr(scheduler, "base_lrs") and i < len(scheduler.base_lrs):
                            scheduler.base_lrs[i] = max(float(scheduler.base_lrs[i]) * float(args.lr_drop_factor), float(args.min_lr))
                    downgrade_events += 1
                    consecutive_nonfinite = 0
                continue

            scaler.scale(loss_to_backprop).backward()

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_finite = all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.grad is not None
                )
                if not grad_finite:
                    bad_steps += 1
                    total_bad_steps += 1
                    consecutive_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                consecutive_nonfinite = 0

            losses.append(total_loss.item())

            if main_proc and step % 30 == 0:
                print(f"[Epoch {epoch + 1}/{args.epochs}] step {step}/{len(train_loader)} "
                      f"loss={total_loss.item():.4f} seg={seg_loss.item():.4f} "
                      f"inside={in_loss.item():.4f} grade={grade_loss.item():.4f} "
                      f"boundary={boundary_loss_term.item():.4f} beta={beta:.1f} lam_grade={lambda_grade:.3f}")

        scheduler.step()

        # ── 评估与保存（仅 rank-0 执行）──────────────────────────
        ddp_barrier()   # 等所有进程跑完当前 epoch
        if main_proc:
            train_loss = float(np.mean(losses)) if losses else 0.0
            metrics = evaluate(model, val_loader, device, args.num_classes, len(grade_values), grade_bounds)

            score = metrics["mIoU"]
            if args.use_grade and "grade_qwk_ratio" in metrics:
                score += float(args.score_ratio_weight) * metrics["grade_qwk_ratio"]
                score += float(args.score_head_weight) * metrics.get("grade_qwk_head", 0.0)

            append_epoch_metrics_csv(epoch_metrics_csv, args.ablation_mode, epoch + 1, train_loss,
                                     optimizer.param_groups[0]["lr"], bad_steps, amp_enabled,
                                     downgrade_events, metrics)

            raw_model = model.module
            state_dict_to_save = (raw_model.get_trainable_state_dict()
                                  if hasattr(raw_model, "get_trainable_state_dict")
                                  else raw_model.state_dict())
            ckpt = {
                "epoch": epoch + 1,
                "model": state_dict_to_save,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
                "args": vars(args),
            }

            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                torch.save(ckpt, os.path.join(args.save_path, "soybean_dm_best.pth"))
                print(f"[Best] epoch={best_epoch}, score={best_score:.4f}")

            if (epoch + 1) % 20 == 0:
                torch.save(ckpt, os.path.join(args.save_path, f"soybean_dm_epoch{epoch + 1}.pth"))
            torch.save(ckpt, os.path.join(args.save_path, "soybean_dm_last.pth"))
            print(f"Epoch {epoch + 1} done | lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"mIoU={metrics['mIoU']:.4f} | elapsed={format_seconds(time.time() - start_time)}")
        # ────────────────────────────────────────────────────────

    # ── 最终评估（仅 rank-0）────────────────────────────────────
    if main_proc:
        best_ckpt_path = os.path.join(args.save_path, "soybean_dm_best.pth")
        if os.path.exists(best_ckpt_path):
            model.module.load_state_dict(
                torch.load(best_ckpt_path, map_location=device)["model"], strict=False)
        final_metrics, final_details = evaluate(
            model, val_loader, device, args.num_classes, len(grade_values), grade_bounds, return_details=True)
        save_final_validation_artifacts(final_metrics, final_details, grade_values, args.save_path)
        export_prediction_csv(
            model, val_loader, device,
            os.path.join(args.save_path, args.pred_csv_name),
            args.leaf_class_id, disease_class_ids, grade_values, grade_bounds)
        print("Training finished successfully.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
