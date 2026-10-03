"""Read-only structural probes for the EM-CLIP paper audit.

Run from the repository with ``python -B scripts/audit_emclip_paper.py``.
This uses a small randomly initialized model; its observations are structural,
not estimates of accuracy or substitutes for real CoViAR decoding.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_coviar import CoviarDataSet, sample_gop_indices
from models.emclip import EMCLIP, EMCLIPConfig
from models.emclip_mgse import select_topk_indices


def main(implementation="paper"):
    torch.manual_seed(1024)
    torch.set_num_threads(1)
    config = EMCLIPConfig(
        num_classes=5,
        class_names=["pulling left", "pulling right", "opening", "closing", "turning"],
        candidate_frames=4,
        selected_frames=2,
        input_size=64,
        width=32,
        layers=2,
        heads=4,
        embed_dim=16,
        text_width=32,
        text_layers=2,
        text_heads=4,
        implementation=implementation,
    )
    model = EMCLIP(config).eval()
    i_frames = torch.randn(2, 4, 3, 64, 64)
    residuals = torch.randn_like(i_frames)
    motion = torch.randn(2, 4, 2, 64, 64)
    valid = torch.ones(2, 4, dtype=torch.bool)
    labels = torch.tensor([0, 1])
    results = {"scope": "synthetic CPU structural probes; no pretrained accuracy claim"}

    # Jointly reorder GOP tuples. Within-GOP MV/R direction stays unchanged;
    # this is not a physical reversal and re-encoding of the underlying video.
    permutation = torch.tensor([3, 1, 0, 2])
    with torch.no_grad():
        reference = model.melsc(i_frames, residuals)["video_features"]
        reordered = model.melsc(i_frames[:, permutation], residuals[:, permutation])["video_features"]
        reversed_order = model.melsc(i_frames.flip(1), residuals.flip(1))["video_features"]
        zero_residual = model.melsc(i_frames, torch.zeros_like(residuals))["video_features"]
    results["gop_order"] = {
        "permutation_max_abs_difference": float((reference - reordered).abs().max()),
        "reverse_order_max_abs_difference": float((reference - reversed_order).abs().max()),
        "zero_residual_max_abs_difference": float((reference - zero_residual).abs().max()),
    }

    # Eq. (13) obtains K_R/V_R from Eq. (16). Capture the actual pre-MHA
    # features of the same layer, before each attention's own W_K/W_V.
    captured = {}

    def capture_gs(module, inputs):
        captured["gs_residual_kv"] = inputs[1].detach()

    def capture_lm(module, inputs):
        captured["lm_residual_kv"] = inputs[1].detach()

    hooks = [
        model.melsc.gs_attn[1].register_forward_pre_hook(capture_gs),
        model.melsc.lm_attn[1].register_forward_pre_hook(capture_lm),
    ]
    with torch.no_grad():
        model.melsc(i_frames, residuals)
    for hook in hooks:
        hook.remove()
    results["shared_residual_feature_eq16"] = {
        "shared_eq16_input": implementation == "paper",
        "pre_attention_kv_max_abs_difference": float(
            (captured["gs_residual_kv"] - captured["lm_residual_kv"]).abs().max()
        ),
    }

    # An explicit class-bank counterexample: opposite category descriptions
    # prefer opposite frames, yet their mean has no temporal preference.
    opposed_motion = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]]) / (2.0 ** 0.5)
    opposed_tokens = opposed_motion[0, :, None, :]
    opposed_mask = torch.ones(2, 1, dtype=torch.bool)
    opposed_valid = torch.ones(1, 2, dtype=torch.bool)
    bank_saliency = model.mgse._class_bank_saliency(
        opposed_motion, opposed_tokens, opposed_mask, opposed_valid
    )
    category_saliency = model.mgse._ground_truth_saliency(
        opposed_motion, opposed_tokens, opposed_mask, torch.tensor([0]), opposed_valid
    )
    results["class_bank_mean_counterexample"] = {
        "class_bank_saliency": bank_saliency.tolist(),
        "single_category_saliency": category_saliency.tolist(),
        "interpretation": "possible cancellation, not a measurement on trained SSV2 features",
    }

    # Short videos repeat real GOPs. top-k operates on candidate slots, not
    # unique GOP IDs, and all temporal views follow the same repeat branch.
    candidates, mask, _ = sample_gop_indices(96, 16, 12, False, gop_count=8,
                                            deduplicate_candidates=implementation == "paper")
    gop_ids = torch.tensor(candidates)
    slot_saliency = (gop_ids.float() + 1.0).unsqueeze(0)
    selected = select_topk_indices(slot_saliency, mask.unsqueeze(0), 8)[0]
    views = [
        sample_gop_indices(96, 16, 12, False, temporal_view=view, num_temporal_views=4, gop_count=8)[0]
        for view in range(4)
    ]
    results["short_video_sampling"] = {
        "candidate_gop_indices": candidates,
        "selected_candidate_indices": selected.tolist(),
        "selected_gop_ids": gop_ids[selected].tolist(),
        "unique_selected_gops": int(gop_ids[selected].unique().numel()),
        "distinct_temporal_views": len({tuple(view) for view in views}),
    }

    # Use the actual augmentation and __getitem__ label handling without
    # claiming a native decode. Force only the horizontal flip random draw.
    dataset = object.__new__(CoviarDataSet)
    dataset.input_size = 4
    dataset.random_sample = True
    dataset.horizontal_flip = implementation == "legacy"
    dataset.items = [SimpleNamespace(label=86)]
    image = torch.arange(48).reshape(1, 3, 4, 4).float()
    mv = torch.ones(1, 2, 4, 4)

    def synthetic_view(item):
        transformed_i, transformed_mv, transformed_r = dataset._augment(image, mv, image)
        return {"i_frames": transformed_i, "motion_vectors": transformed_mv, "residuals": transformed_r}

    dataset._load_view = synthetic_view
    with patch("dataset_coviar.random.random", return_value=0.0):
        flipped = dataset[0]
    results["ssv2_horizontal_flip"] = {
        "input_label": 86,
        "output_label": int(flipped["label"]),
        "direction_pair_other_label": 87,
        "horizontal_flip_applied": bool(torch.equal(flipped["i_frames"], image.flip(-1))),
        "mv_x_after_flip": float(flipped["motion_vectors"][0, 0, 0, 0]),
    }

    dataset.residual_scale = 255.0
    dataset.residual_channel_order = "rgb" if implementation == "paper" else "bgr"
    bgr_sample = np.array([[[10, 20, 30]]], dtype=np.uint8)
    results["native_bgr_channel_handling"] = {
        "raw_channels": [10, 20, 30],
        "i_channels_times_255": (dataset._prepare_iframe(bgr_sample)[:, 0, 0] * 255).tolist(),
        "r_channels_times_255": (dataset._prepare_residual(bgr_sample.astype(np.int32), (1, 1))[:, 0, 0] * 255).tolist(),
    }

    # L_ME has no gradient through integer top-k; L_MG provides the MGSE path.
    model.train()
    output = model(i_frames, motion, residuals, labels=labels, valid_mask=valid)
    mgse_parameters = list(model.mgse.parameters())
    ce_gradients = torch.autograd.grad(
        output["loss_me"], mgse_parameters, allow_unused=True, retain_graph=True
    )
    output["loss"].backward()
    unused = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    nonfinite = [
        name for name, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    results["loss_and_backward"] = {
        "loss": float(output["loss"].detach()),
        "loss_mg": float(output["loss_mg"].detach()),
        "loss_me": float(output["loss_me"].detach()),
        "total_matches_weighted_components": bool(torch.allclose(
            output["loss"], output["loss_mg"] + output["loss_me"]
        )),
        "ce_reaches_mgse_parameter_count": sum(gradient is not None for gradient in ce_gradients),
        "unused_trainable_parameters": unused,
        "nonfinite_gradients": nonfinite,
        "unpretrained_effective_classification_temperature": 1.0 / float(output["classification_logit_scale"]),
        "mgse_temperature": config.mgse_temperature,
    }
    if unused or nonfinite or not torch.isfinite(output["loss"]):
        raise RuntimeError("Synthetic forward/backward failed: %s" % results["loss_and_backward"])
    print(json.dumps(results, ensure_ascii=False, indent=2))


def _ddp_worker(rank, store_uri, implementation, variant):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=2)
    try:
        torch.manual_seed(1024)
        config = EMCLIPConfig(
            num_classes=3,
            class_names=["pulling left", "pulling right", "opening"],
            candidate_frames=4,
            selected_frames=2,
            input_size=64,
            width=32,
            layers=2,
            heads=4,
            embed_dim=16,
            text_width=32,
            text_heads=4,
            text_layers=2,
            implementation=implementation,
            emclip_variant=variant,
        )
        model = torch.nn.parallel.DistributedDataParallel(
            EMCLIP(config), find_unused_parameters=False
        )
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        torch.manual_seed(1024 + rank)
        losses = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(
                torch.randn(2, 4, 3, 64, 64),
                torch.randn(2, 4, 2, 64, 64) if variant == "emclip" else None,
                torch.randn(2, 4, 3, 64, 64),
                labels=torch.tensor([0, rank + 1]),
                training_mode=True,
            )
            output["loss"].backward()
            bad = [
                name for name, p in model.named_parameters()
                if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
            ]
            if bad:
                raise RuntimeError("DDP missing/nonfinite gradients: %s" % bad)
            optimizer.step()
            losses.append(float(output["loss"].detach()))
        parameters = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        gathered = [torch.empty_like(parameters) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, parameters)
        difference = max(float((parameters - p).abs().max()) for p in gathered)
        if difference != 0.0:
            raise RuntimeError("DDP rank parameters diverged: %s" % difference)
        if rank == 0:
            print(json.dumps({
                "world_size": 2,
                "backend": "CPU Gloo / FileStore",
                "two_optimizer_steps": "PASS",
                "losses_rank0": losses,
                "max_parameter_difference_across_ranks": difference,
                "find_unused_parameters": False,
                "implementation": implementation,
                "variant": variant,
            }, indent=2))
    finally:
        dist.destroy_process_group()


def ddp_smoke(implementation="paper", variant="emclip"):
    import tempfile
    import torch.multiprocessing as mp

    # FileStore avoids depending on libuv-enabled TCPStore in Windows wheels.
    with tempfile.TemporaryDirectory(prefix="emclip_audit_ddp_") as directory:
        store_uri = (Path(directory) / "store").as_uri()
        mp.spawn(_ddp_worker, args=(store_uri, implementation, variant), nprocs=2, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddp-smoke", action="store_true", help="Run only two synthetic CPU Gloo optimizer steps.")
    parser.add_argument("--implementation", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--ddp-variant", choices=["emclip", "diamond"], default="emclip")
    arguments = parser.parse_args()
    if arguments.ddp_smoke:
        ddp_smoke(arguments.implementation, arguments.ddp_variant)
    else:
        main(arguments.implementation)
