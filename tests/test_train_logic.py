import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

model_last_stub = types.ModuleType("model.model_last")


class _UnusedDMNetWithClustering:
    pass


model_last_stub.DMNetWithClustering = _UnusedDMNetWithClustering
sys.modules.setdefault("model.model_last", model_last_stub)

dataset_stub = types.ModuleType("dataset")


class _UnusedFullDataset:
    pass


dataset_stub.FullDataset = _UnusedFullDataset
sys.modules.setdefault("dataset", dataset_stub)

from train.train_ddp import boundary_target_from_mask, consistency_correction_loss


def test_boundary_target_uses_only_disease_boundary_when_disease_ids_are_given():
    mask = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 2, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
        dtype=torch.long,
    )

    target = boundary_target_from_mask(mask, disease_class_ids=(2,))

    expected = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    expected[0, 0, 2, 2] = 1.0
    expected[0, 0, 1, 2] = 1.0
    expected[0, 0, 3, 2] = 1.0
    expected[0, 0, 2, 1] = 1.0
    expected[0, 0, 2, 3] = 1.0

    assert torch.equal(target, expected)


def test_consistency_correction_loss_trains_corrected_output_path():
    fusion_bias = nn.Parameter(torch.tensor([0.0]))
    grade_probs_head = torch.tensor(
        [[0.65, 0.15, 0.08, 0.06, 0.04, 0.02]],
        dtype=torch.float32,
        requires_grad=True,
    )
    ratio_grade_probs = torch.tensor(
        [[0.02, 0.04, 0.08, 0.18, 0.50, 0.18]],
        dtype=torch.float32,
    )
    grade_values_norm = torch.tensor([0, 1, 3, 5, 7, 9], dtype=torch.float32) / 9.0
    grade_score_head = (grade_probs_head * grade_values_norm.unsqueeze(0)).sum(dim=1)
    ratio_grade_score = (ratio_grade_probs * grade_values_norm.unsqueeze(0)).sum(dim=1)
    weight = torch.sigmoid(fusion_bias).view(1, 1)
    corrected_probs = weight * grade_probs_head + (1.0 - weight) * ratio_grade_probs
    corrected_score = (corrected_probs * grade_values_norm.unsqueeze(0)).sum(dim=1)

    aux = {
        "grade_probs_head": grade_probs_head,
        "grade_score_head": grade_score_head,
        "ratio_grade_probs": ratio_grade_probs,
        "ratio_grade_score": ratio_grade_score,
        "grade_probs": corrected_probs,
        "grade_score": corrected_score,
        "inconsistency_score": (grade_score_head - ratio_grade_score).abs(),
    }

    loss = consistency_correction_loss(aux)
    loss.backward()

    assert fusion_bias.grad is not None
    assert fusion_bias.grad.abs().item() > 0.0
