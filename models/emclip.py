import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from losses.emclip_loss import motion_text_kl_loss
from .clip_checkpoint import (
    adapt_visual_positional_embedding,
    extract_text_state,
    extract_visual_state,
    load_clip_source_checkpoint,
    load_submodule_with_audit,
)
from .clip_tokenizer import OpenAIClipBPETokenizer
from .emclip_layers import LayerNorm, TransformerBlock
from .emclip_melsc import MotionEmbeddedLongTermSpatiotemporalCorrelation
from .emclip_mgse import (
    MotionGuidedSaliencyExtraction,
    build_mv_patch_embed_weight,
    gather_temporal,
)


def _is_rank_zero():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


@dataclass
class EMCLIPConfig:
    num_classes: int
    class_names: List[str]
    backbone: str = "ViT-B/16"
    candidate_frames: int = 16
    selected_frames: int = 8
    input_size: int = 256
    gop_size: int = 12
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    embed_dim: int = 512
    text_width: int = 512
    text_heads: int = 8
    text_layers: int = 12
    text_context_length: int = 77
    vocab_size: int = 49408
    prompt_templates: List[str] = field(default_factory=lambda: ["a photo of a {}"])
    mgse_temperature: float = 0.01
    mgse_text_mode: str = "class_bank"
    mgse_class_aggregation: str = "mean"
    motion_pooling: str = "saliency"
    lambda_mg: float = 1.0
    lambda_me: float = 1.0
    emclip_variant: str = "emclip"
    emclip_train_mode: str = "full"
    temporal_aggregator_layers: int = 1
    dropout: float = 0.0
    allow_mgse_label_leakage_for_diagnostic: bool = False
    debug_shapes: bool = False
    clip_checkpoint: Optional[str] = None
    clip_bpe_path: Optional[str] = None

    def __post_init__(self):
        if self.width % self.heads != 0:
            raise ValueError("visual width must be divisible by visual heads.")
        if self.text_width % self.text_heads != 0:
            raise ValueError("text_width must be divisible by text_heads.")
        if self.num_classes != len(self.class_names):
            raise ValueError(
                "num_classes=%d but class_names has %d entries."
                % (self.num_classes, len(self.class_names))
            )
        if self.selected_frames > self.candidate_frames:
            raise ValueError("selected_frames cannot exceed candidate_frames.")
        variant = self.emclip_variant.lower()
        if variant in ("emclip_diamond", "diamond"):
            self.emclip_variant = "diamond"
        elif variant in ("emclip", "full"):
            self.emclip_variant = "emclip"
        else:
            raise ValueError("emclip_variant must be 'emclip' or 'diamond'.")


class PromptTextEncoder(nn.Module):
    """CLIP-style text encoder for class EOT and label-word token features."""

    def __init__(
        self,
        class_names,
        width,
        layers,
        heads,
        embed_dim,
        context_length=77,
        vocab_size=49408,
        prompt_templates=None,
        dropout=0.0,
        bpe_path=None,
    ):
        super().__init__()
        self.class_names = list(class_names)
        self.context_length = context_length
        self.prompt_templates = prompt_templates or ["a photo of a {}"]
        for template in self.prompt_templates:
            if "{}" not in template:
                raise ValueError("prompt template must contain '{}': %s" % template)
        self.tokenizer = OpenAIClipBPETokenizer(bpe_path=bpe_path, context_length=context_length)
        if vocab_size != self.tokenizer.vocab_size:
            raise ValueError(
                "text vocab_size=%d does not match OpenAI CLIP tokenizer vocab_size=%d"
                % (vocab_size, self.tokenizer.vocab_size)
            )
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads, dropout=dropout) for _ in range(layers)])
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        self._eval_cache = None
        self._init_weights(width)

    def _init_weights(self, width):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=width ** -0.5)

    def _tokenize_templates(self, device):
        token_rows = []
        label_masks = []
        eot_positions = []
        class_ids = []
        template_ids = []
        for class_idx, name in enumerate(self.class_names):
            for template_idx, template in enumerate(self.prompt_templates):
                token_ids, label_span, eot = self.tokenizer.encode_prompt(template, name)
                tokens = torch.zeros(self.context_length, dtype=torch.long)
                tokens[:len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
                label_mask = torch.zeros(self.context_length, dtype=torch.bool)
                label_mask[label_span[0]:label_span[1]] = True
                token_rows.append(tokens)
                label_masks.append(label_mask)
                eot_positions.append(eot)
                class_ids.append(class_idx)
                template_ids.append(template_idx)
        return (
            torch.stack(token_rows).to(device),
            torch.stack(label_masks).to(device),
            torch.tensor(eot_positions, dtype=torch.long, device=device),
            torch.tensor(class_ids, dtype=torch.long, device=device),
            torch.tensor(template_ids, dtype=torch.long, device=device),
        )

    def _encode_tokens(self, tokens):
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding.to(device=x.device, dtype=x.dtype)
        causal_mask = torch.full(
            (self.context_length, self.context_length),
            float("-inf"),
            dtype=x.dtype,
            device=x.device,
        ).triu_(1)
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        x = self.ln_final(x)
        return x

    def encode_class_prompts(self, training_mode=True):
        device = self.text_projection.device
        if training_mode:
            self.clear_cache()
        if (not training_mode) and self._eval_cache is not None and self._eval_cache[0].device == device:
            return self._eval_cache
        tokens, label_masks, eot_positions, class_ids, _ = self._tokenize_templates(device)
        token_features = self._encode_tokens(tokens)
        eot_features = token_features[torch.arange(tokens.size(0), device=device), eot_positions]
        eot_features = eot_features @ self.text_projection.to(device=device, dtype=eot_features.dtype)

        C = len(self.class_names)
        D = eot_features.size(-1)
        class_features = torch.zeros(C, D, dtype=eot_features.dtype, device=device)
        class_features.index_add_(0, class_ids, eot_features)
        class_features = class_features / float(len(self.prompt_templates))
        class_features = F.normalize(class_features.float(), dim=-1)

        projected_tokens = token_features @ self.text_projection.to(device=device, dtype=token_features.dtype)
        max_label_tokens = int(label_masks.sum(dim=1).max().item())
        max_label_tokens = max(1, max_label_tokens)
        class_token_features = torch.zeros(C, max_label_tokens, D, dtype=projected_tokens.dtype, device=device)
        class_token_counts = torch.zeros(C, max_label_tokens, dtype=torch.float32, device=device)
        for row in range(tokens.size(0)):
            cls = int(class_ids[row].item())
            selected = projected_tokens[row][label_masks[row]]
            if selected.numel() == 0:
                selected = eot_features[row:row + 1]
            count = min(max_label_tokens, selected.size(0))
            class_token_features[cls, :count] += selected[:count]
            class_token_counts[cls, :count] += 1.0
        class_token_mask = class_token_counts > 0
        class_token_features = class_token_features / class_token_counts.clamp_min(1.0).unsqueeze(-1)
        class_token_features = F.normalize(class_token_features.float(), dim=-1)
        result = (class_features, class_token_features, class_token_mask)
        if not training_mode:
            self._eval_cache = result
        return result

    def clear_cache(self):
        self._eval_cache = None

    def train(self, mode=True):
        if mode:
            self.clear_cache()
        return super().train(mode)


def _select_diamond_indices(valid_mask, selected_frames):
    assert valid_mask.ndim == 2, "valid_mask must be [B, T]."
    B, T = valid_mask.shape
    if selected_frames > T:
        raise ValueError("selected_frames cannot exceed candidate count.")
    rows = []
    for b in range(B):
        valid = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError("diamond sampling needs at least one valid candidate.")
        if valid.numel() >= selected_frames:
            pos = torch.linspace(0, valid.numel() - 1, selected_frames, device=valid.device)
            rows.append(valid[pos.round().long()])
        else:
            repeat = valid[-1:].repeat(selected_frames - valid.numel())
            rows.append(torch.cat([valid, repeat], dim=0))
    return torch.stack(rows, dim=0)


class EMCLIP(nn.Module):
    """End-to-end EM-CLIP with MGSE, MELSC, L_MG, and L_ME."""

    def __init__(self, config: EMCLIPConfig):
        super().__init__()
        self.config = config
        self.text_encoder = PromptTextEncoder(
            class_names=config.class_names,
            width=config.text_width,
            layers=config.text_layers,
            heads=config.text_heads,
            embed_dim=config.embed_dim,
            context_length=config.text_context_length,
            vocab_size=config.vocab_size,
            prompt_templates=config.prompt_templates,
            dropout=config.dropout,
            bpe_path=config.clip_bpe_path,
        )
        self.use_mgse = config.emclip_variant == "emclip"
        self.mgse = MotionGuidedSaliencyExtraction(
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            output_dim=config.embed_dim,
            input_resolution=config.input_size,
            patch_size=config.patch_size,
            selected_frames=config.selected_frames,
            temperature=config.mgse_temperature,
            text_mode=config.mgse_text_mode,
            class_aggregation=config.mgse_class_aggregation,
            motion_pooling=config.motion_pooling,
            dropout=config.dropout,
            allow_label_leakage_for_diagnostic=config.allow_mgse_label_leakage_for_diagnostic,
        ) if self.use_mgse else None
        self.melsc = MotionEmbeddedLongTermSpatiotemporalCorrelation(
            width=config.width,
            layers=config.layers,
            heads=config.heads,
            output_dim=config.embed_dim,
            input_resolution=config.input_size,
            patch_size=config.patch_size,
            temporal_aggregator_layers=config.temporal_aggregator_layers,
            dropout=config.dropout,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.pretrained_audit = None
        if config.clip_checkpoint:
            self.load_clip_checkpoint(config.clip_checkpoint)
        self.apply_train_mode(config.emclip_train_mode)

    def load_clip_checkpoint(self, path):
        """Initialize I/R/MV/Text branches from an original CLIP checkpoint."""
        source, checkpoint_type, checkpoint_path = load_clip_source_checkpoint(path)
        if _is_rank_zero():
            print("[emclip][pretrained] checkpoint_type=%s" % checkpoint_type, flush=True)
            print("[emclip][pretrained] checkpoint_path=%s" % checkpoint_path, flush=True)
            print(
                "[emclip][pretrained] first_source_keys=%s" % list(source.keys())[:20],
                flush=True,
            )

        visual_source = extract_visual_state(source)
        text_source = extract_text_state(source)
        visual_required = {
            "conv1.weight",
            "class_embedding",
            "positional_embedding",
            "ln_pre.weight",
            "ln_pre.bias",
            "ln_post.weight",
            "ln_post.bias",
            "proj",
        }
        text_required = {
            "token_embedding.weight",
            "positional_embedding",
            "ln_final.weight",
            "ln_final.bias",
            "text_projection",
        }
        reports = {}
        grids = {}
        for name, module in (
            ("I_encoder", self.melsc.i_encoder),
            ("R_encoder", self.melsc.r_encoder),
        ):
            branch_source, grid_pair = adapt_visual_positional_embedding(module, visual_source)
            grids[name] = grid_pair
            reports[name] = load_submodule_with_audit(
                module,
                branch_source,
                name,
                minimum_coverage=0.99,
                required_keys=visual_required,
            )

        if self.mgse is not None:
            mv_source, grid_pair = adapt_visual_positional_embedding(
                self.mgse.motion_encoder,
                visual_source,
            )
            grids["MV_encoder"] = grid_pair
            rgb_conv = visual_source.get("conv1.weight")
            if rgb_conv is None:
                raise RuntimeError("CLIP visual checkpoint is missing conv1.weight for MV initialization")
            mv_source["conv1.weight"] = build_mv_patch_embed_weight(rgb_conv, scale=True)
            reports["MV_encoder"] = load_submodule_with_audit(
                self.mgse.motion_encoder,
                mv_source,
                "MV_encoder_excluding_conv1",
                excluded_from_coverage={"conv1.weight"},
                minimum_coverage=0.99,
                required_keys=visual_required,
            )

        reports["Text_encoder"] = load_submodule_with_audit(
            self.text_encoder,
            text_source,
            "Text_encoder",
            minimum_coverage=0.99,
            required_keys=text_required,
        )
        if "logit_scale" not in source:
            raise RuntimeError("CLIP checkpoint is missing required text parameter 'logit_scale'")
        if tuple(source["logit_scale"].shape) != tuple(self.logit_scale.shape):
            raise RuntimeError(
                "CLIP logit_scale shape mismatch: source=%s target=%s"
                % (tuple(source["logit_scale"].shape), tuple(self.logit_scale.shape))
            )
        with torch.no_grad():
            self.logit_scale.copy_(source["logit_scale"])

        # ``load_state_dict`` copies into each branch.  Assert this invariant so
        # I and residual can subsequently train independently.
        i_parameters = dict(self.melsc.i_encoder.named_parameters())
        r_parameters = dict(self.melsc.r_encoder.named_parameters())
        for key in sorted(set(i_parameters).intersection(r_parameters)):
            i_parameter = i_parameters[key]
            r_parameter = r_parameters[key]
            if i_parameter is r_parameter or i_parameter.data_ptr() == r_parameter.data_ptr():
                raise RuntimeError("I/R encoders unexpectedly share parameter storage for %s" % key)

        self.pretrained_audit = {
            "checkpoint_type": checkpoint_type,
            "checkpoint_path": checkpoint_path,
            "position_grids": grids,
            "branches": {name: report.to_dict() for name, report in reports.items()},
        }
        if _is_rank_zero():
            for name, report in reports.items():
                grid_pair = grids.get(name)
                if grid_pair is not None:
                    print(
                        "[emclip][pretrained] %s_position_grid=%s->%s"
                        % (name, grid_pair[0], grid_pair[1]),
                        flush=True,
                    )
                print(
                    "[emclip][pretrained] %s source=%d loaded=%d numel=%d/%d "
                    "coverage=%.2f%% missing=%s unexpected=%s shape_mismatch=%s"
                    % (
                        name,
                        report.source_tensor_count,
                        report.loaded_tensor_count,
                        report.loaded_parameter_numel,
                        report.total_parameter_numel,
                        100.0 * report.coverage,
                        report.missing_keys,
                        report.unexpected_keys,
                        report.shape_mismatch_keys,
                    ),
                    flush=True,
                )
            print(
                "[emclip][pretrained] logit_scale=%.8f"
                % float(self.logit_scale.detach().float().cpu()),
                flush=True,
            )
        return self.pretrained_audit

    def apply_train_mode(self, mode):
        mode = mode.lower()
        if mode not in ("full", "freeze_text", "freeze_clip"):
            raise ValueError("emclip_train_mode must be full, freeze_text, or freeze_clip.")
        for parameter in self.parameters():
            parameter.requires_grad = True
        if mode == "freeze_text":
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False
        elif mode == "freeze_clip":
            for name, parameter in self.named_parameters():
                parameter.requires_grad = False
                train_new = (
                    "mgse.feature_ln" in name or
                    "mgse.motion_encoder.proj" in name or
                    "melsc.gs_" in name or
                    "melsc.lm_" in name or
                    "melsc.temporal_blocks" in name or
                    "melsc.final_ln" in name
                )
                if train_new:
                    parameter.requires_grad = True
        self._freeze_parameters_without_loss_path()

    def _freeze_module(self, module):
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _freeze_parameters_without_loss_path(self):
        self.melsc.freeze_unused_parameters()
        if self.use_mgse and self.config.motion_pooling == "mean":
            self._freeze_module(self.mgse.feature_ln)
        if self.use_mgse and float(self.config.lambda_mg) == 0.0:
            self._freeze_module(self.mgse)
        if float(self.config.lambda_me) == 0.0:
            self._freeze_module(self.melsc)
            self.logit_scale.requires_grad = False

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def _validate_inputs(self, i_frames, motion_vectors, residuals):
        assert i_frames.ndim == 5, "I frames must be [B, T, 3, H, W]."
        assert residuals.ndim == 5, "Residuals must be [B, T, 3, H, W]."
        assert i_frames.size(2) == 3 and residuals.size(2) == 3
        assert i_frames.shape[:2] == residuals.shape[:2]
        if self.use_mgse:
            assert motion_vectors is not None, "EM-CLIP requires motion_vectors [B, T, 2, H, W]."
            assert motion_vectors.ndim == 5 and motion_vectors.size(2) == 2
            assert motion_vectors.shape[:2] == i_frames.shape[:2]

    def forward(
        self,
        i_frames,
        motion_vectors,
        residuals,
        labels=None,
        mgse_labels=None,
        valid_mask=None,
        training_mode=None,
    ):
        self._validate_inputs(i_frames, motion_vectors, residuals)
        B, T = i_frames.shape[:2]
        if training_mode is None:
            training_mode = self.training
        if valid_mask is None:
            valid_mask = torch.ones(B, T, dtype=torch.bool, device=i_frames.device)
        else:
            valid_mask = valid_mask.to(device=i_frames.device, dtype=torch.bool)
        if labels is not None:
            labels = labels.to(device=i_frames.device, dtype=torch.long)
        if mgse_labels is None:
            mgse_labels = labels
        elif mgse_labels is not None:
            mgse_labels = mgse_labels.to(device=i_frames.device, dtype=torch.long)

        class_text_features, class_token_features, class_token_mask = self.text_encoder.encode_class_prompts(
            training_mode=training_mode
        )
        class_text_features = class_text_features.to(device=i_frames.device)
        class_token_features = class_token_features.to(device=i_frames.device)
        class_token_mask = class_token_mask.to(device=i_frames.device)

        if self.use_mgse:
            mgse_out = self.mgse(
                motion_vectors=motion_vectors,
                class_text_features=class_text_features,
                class_token_features=class_token_features,
                class_token_mask=class_token_mask,
                labels=mgse_labels,
                valid_mask=valid_mask,
                training_mode=training_mode,
            )
            selected_indices = mgse_out["selected_indices"]
            saliency = mgse_out["saliency"]
            motion_frame_features = mgse_out["motion_frame_features"]
        else:
            selected_indices = _select_diamond_indices(valid_mask, self.config.selected_frames)
            saliency = torch.zeros(B, T, dtype=torch.float32, device=i_frames.device)
            motion_frame_features = None
            mgse_out = {
                "motion_video_features": None,
                "mg_logits_mv2text": None,
                "mg_logits_text2mv": None,
            }

        selected_i = gather_temporal(i_frames, selected_indices)
        selected_r = gather_temporal(residuals, selected_indices)
        assert selected_i.shape[:2] == (B, self.config.selected_frames)
        melsc_out = self.melsc(selected_i, selected_r)
        video_features = melsc_out["video_features"]
        class_text_features = F.normalize(class_text_features.float(), dim=-1)
        logit_scale = self.logit_scale.float().exp().clamp(max=100.0)
        logits = logit_scale * (video_features.float() @ class_text_features.t())
        loss_me = None
        loss_mg = logits.new_zeros(())
        loss_mg_mv2text = logits.new_zeros(())
        loss_mg_text2mv = logits.new_zeros(())
        if labels is not None:
            loss_me = F.cross_entropy(logits.float(), labels)
            if self.use_mgse:
                mg_loss = motion_text_kl_loss(
                    motion_frame_features=motion_frame_features,
                    saliency=saliency,
                    class_text_features=class_text_features,
                    labels=labels,
                    valid_mask=valid_mask,
                    pooling=self.config.motion_pooling,
                    temperature=self.config.mgse_temperature,
                )
                loss_mg = mg_loss["loss_mg"]
                loss_mg_mv2text = mg_loss["loss_mg_mv2text"]
                loss_mg_text2mv = mg_loss["loss_mg_text2mv"]
            loss = self.config.lambda_me * loss_me + self.config.lambda_mg * loss_mg
        else:
            loss = None
        if loss is not None and not torch.isfinite(loss):
            raise FloatingPointError("EM-CLIP total loss is non-finite.")
        entropy = -(saliency.clamp_min(1e-8) * saliency.clamp_min(1e-8).log()).sum(dim=1).mean()
        return {
            "loss": loss,
            "loss_mg": loss_mg,
            "loss_mg_mv2text": loss_mg_mv2text,
            "loss_mg_text2mv": loss_mg_text2mv,
            "loss_me": loss_me,
            "logits": logits,
            "video_features": video_features,
            "class_text_features": class_text_features,
            "selected_indices": selected_indices,
            "selected_i_frames": selected_i,
            "selected_residuals": selected_r,
            "saliency": saliency,
            "mgse_saliency_entropy": entropy,
            "motion_frame_features": motion_frame_features,
            "melsc_debug": melsc_out["debug"],
            **mgse_out,
        }


def build_emclip_config_from_args(args, class_names):
    variant = getattr(args, "emclip_variant", "emclip")
    return EMCLIPConfig(
        num_classes=len(class_names),
        class_names=class_names,
        candidate_frames=args.candidate_frames,
        selected_frames=args.selected_frames,
        input_size=args.input_size,
        gop_size=args.gop_size,
        mgse_temperature=args.mgse_temperature,
        mgse_text_mode=args.mgse_text_mode,
        mgse_class_aggregation=args.mgse_class_aggregation,
        motion_pooling=args.motion_pooling,
        lambda_mg=args.lambda_mg,
        lambda_me=args.lambda_me,
        emclip_variant=variant,
        emclip_train_mode=args.emclip_train_mode,
        temporal_aggregator_layers=args.temporal_aggregator_layers,
        allow_mgse_label_leakage_for_diagnostic=args.allow_mgse_label_leakage_for_diagnostic,
        debug_shapes=args.debug_shapes,
        clip_checkpoint=args.clip_checkpoint,
        clip_bpe_path=args.clip_bpe_path,
    )


def add_emclip_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", default="emclip_b16")
    parser.add_argument("--emclip-variant", default="emclip", choices=["emclip", "diamond", "emclip_diamond"])
    parser.add_argument("--candidate-frames", type=int, default=16)
    parser.add_argument("--selected-frames", type=int, default=8)
    parser.add_argument("--gop-size", type=int, default=12)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--mgse-temperature", type=float, default=0.01)
    parser.add_argument("--mgse-text-mode", default="class_bank", choices=["class_bank", "ground_truth", "predicted_class"])
    parser.add_argument("--mgse-class-aggregation", default="mean", choices=["mean", "max", "logsumexp"])
    parser.add_argument("--motion-pooling", default="saliency", choices=["saliency", "mean"])
    parser.add_argument("--lambda-mg", type=float, default=1.0)
    parser.add_argument("--lambda-me", type=float, default=1.0)
    parser.add_argument("--emclip-train-mode", default="full", choices=["full", "freeze_text", "freeze_clip"])
    parser.add_argument("--temporal-aggregator-layers", type=int, default=1)
    parser.add_argument("--allow-mgse-label-leakage-for-diagnostic", action="store_true")
    parser.add_argument("--debug-shapes", action="store_true")
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument("--clip-bpe-path", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--pretrained-audit-only", action="store_true")
    return parser
