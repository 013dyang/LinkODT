# -*- coding: utf-8 -*-
"""LinkNex semantic segmentation model factory.

All public LinkNex semantic models consume ONE grayscale channel (B,1,H,W).
DINOv2 ImageNet-pretrained RGB patch-embedding weights are adapted to one
channel by averaging the RGB kernels before microscopy fine-tuning.

Supported architectures
-----------------------
- unet
- dinov2_unet
- dinov2_dpt
- dinov2_dpt_hr
- dinov2_upernet
- dinov2_querymask   (Mask2Former-style query decoder; pure PyTorch)
- segformer_b2       (official Hugging Face SegFormer-B2 backbone; optional)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math


MODEL_SPECS: Dict[str, dict] = {
    "unet": {
        "label": "U-Net",
        "uses_dino": False,
        "recommended": False,
        "advanced": False,
    },
    "dinov2_unet": {
        "label": "DINOv2 + U-Net",
        "uses_dino": True,
        "recommended": False,
        "advanced": False,
    },
    "dinov2_dpt": {
        "label": "DINOv2 + DPT",
        "uses_dino": True,
        "recommended": False,
        "advanced": False,
    },
    "dinov2_dpt_hr": {
        "label": "DINOv2 + DPT-HR  ★ Recommended",
        "uses_dino": True,
        "recommended": True,
        "advanced": False,
    },
    "dinov2_upernet": {
        "label": "DINOv2 + UPerNet",
        "uses_dino": True,
        "recommended": False,
        "advanced": False,
    },
    "dinov2_querymask": {
        "label": "DINOv2 + QueryMask (Mask2Former-style) · Advanced",
        "uses_dino": True,
        "recommended": False,
        "advanced": True,
    },
    "segformer_b2": {
        "label": "SegFormer-B2 · External baseline",
        "uses_dino": False,
        "recommended": False,
        "advanced": False,
    },
}

DINO_BACKBONES: Dict[str, dict] = {
    "dinov2_vits14": {
        "label": "ViT-S/14 · Fast",
        "weight_filename": "dinov2_vits14_pretrain.pth",
        "embed_dim": 384,
    },
    "dinov2_vitb14": {
        "label": "ViT-B/14 · Recommended",
        "weight_filename": "dinov2_vitb14_pretrain.pth",
        "embed_dim": 768,
    },
    "dinov2_vits14_reg": {
        "label": "ViT-S/14 + registers",
        "weight_filename": "dinov2_vits14_reg4_pretrain.pth",
        "embed_dim": 384,
    },
    "dinov2_vitb14_reg": {
        "label": "ViT-B/14 + registers",
        "weight_filename": "dinov2_vitb14_reg4_pretrain.pth",
        "embed_dim": 768,
    },
}


def model_label(key: str) -> str:
    return MODEL_SPECS.get(key, {}).get("label", str(key))


def uses_dino(key: str) -> bool:
    return bool(MODEL_SPECS.get(key, {}).get("uses_dino", False))


def _clean_state_dict(state: dict) -> dict:
    cleaned = {}
    prefixes = (
        "module.", "backbone.", "encoder.", "student_backbone.",
        "teacher_backbone.", "model.",
    )
    for key, value in state.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
                    break
        cleaned[new_key] = value
    return cleaned


def _extract_state_dict(checkpoint) -> dict:
    if isinstance(checkpoint, dict):
        # Raw state dict is the most common encoder-only SSL output.
        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return _clean_state_dict(checkpoint)
        for key in (
            "state_dict", "model", "encoder", "backbone",
            "student_backbone", "teacher_backbone",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _clean_state_dict(value)
    raise ValueError("Could not identify a backbone state_dict in checkpoint.")


def _replace_conv_in_channels(conv, in_channels: int, torch, nn):
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("Expected Conv2d patch projection.")
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        if in_channels == 1 and conv.in_channels == 3:
            new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _convert_dino_to_gray(encoder, torch, nn):
    encoder.patch_embed.proj = _replace_conv_in_channels(
        encoder.patch_embed.proj, 1, torch, nn
    )
    return encoder


def _load_backbone_checkpoint(encoder, path: Path, torch, nn) -> Tuple[List[str], List[str]]:
    path = Path(path)
    raw = torch.load(path, map_location="cpu")
    state = _extract_state_dict(raw)

    patch_key = None
    for key in state:
        if key.endswith("patch_embed.proj.weight"):
            patch_key = key
            break

    # If the SSL checkpoint is already true one-channel, convert architecture
    # before loading. Otherwise load RGB pretrained weights first, then average.
    state_is_gray = (
        patch_key is not None
        and getattr(state[patch_key], "ndim", 0) == 4
        and int(state[patch_key].shape[1]) == 1
    )

    if state_is_gray:
        _convert_dino_to_gray(encoder, torch, nn)

    incompatible = encoder.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    if not state_is_gray:
        _convert_dino_to_gray(encoder, torch, nn)

    # Make obvious architecture mismatches fail early.
    model_keys = len(encoder.state_dict())
    matched = model_keys - len(missing)
    if model_keys and matched < int(0.80 * model_keys):
        raise ValueError(
            f"Backbone checkpoint does not match selected DINOv2 architecture.\n"
            f"Checkpoint: {path}\nMatched keys: {matched}/{model_keys}\n"
            f"Missing: {len(missing)} · Unexpected: {len(unexpected)}"
        )
    return missing, unexpected


def _build_dino_encoder(repo_dir: Path, backbone_name: str, pretrained_path: Optional[Path], torch, nn):
    repo_dir = Path(repo_dir)
    if not (repo_dir / "hubconf.py").exists():
        raise FileNotFoundError(f"DINOv2 local repository is invalid: {repo_dir}")
    if backbone_name not in DINO_BACKBONES:
        raise ValueError(f"Unsupported DINOv2 backbone: {backbone_name}")

    encoder = torch.hub.load(
        str(repo_dir),
        backbone_name,
        source="local",
        pretrained=False,
        verbose=False,
    )

    missing: List[str] = []
    unexpected: List[str] = []
    if pretrained_path is not None:
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"DINOv2 encoder checkpoint not found: {pretrained_path}")
        missing, unexpected = _load_backbone_checkpoint(
            encoder, pretrained_path, torch, nn
        )
    else:
        _convert_dino_to_gray(encoder, torch, nn)

    return encoder, missing, unexpected


def _feature_indices(encoder) -> List[int]:
    n_blocks = len(encoder.blocks)
    if n_blocks < 4:
        return list(range(n_blocks))
    # Approximately 25%, 50%, 75%, 100% depth.
    idx = [
        max(0, round(n_blocks * 0.25) - 1),
        max(0, round(n_blocks * 0.50) - 1),
        max(0, round(n_blocks * 0.75) - 1),
        n_blocks - 1,
    ]
    # preserve order while removing duplicates
    out = []
    for i in idx:
        if i not in out:
            out.append(i)
    while len(out) < 4:
        for i in range(n_blocks):
            if i not in out:
                out.insert(0, i)
                if len(out) == 4:
                    break
    return out[-4:]


def _group_norm(nn, channels: int):
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _make_common_classes(torch, nn, F):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, dilation=1):
            super().__init__()
            pad = dilation
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
                _group_norm(nn, out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                _group_norm(nn, out_ch),
                nn.GELU(),
            )
        def forward(self, x):
            return self.block(x)

    class UpFuse(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.refine = ConvBlock(in_ch + skip_ch, out_ch)
        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.refine(torch.cat([x, skip], dim=1))

    class ResidualFusion(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.res1 = ConvBlock(ch, ch)
            self.res2 = ConvBlock(ch, ch)
        def forward(self, x, residual=None, size=None):
            if residual is not None:
                if residual.shape[-2:] != x.shape[-2:]:
                    residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = x + residual
            x = x + self.res1(x)
            x = x + self.res2(x)
            if size is not None:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return x

    return ConvBlock, UpFuse, ResidualFusion


def _build_plain_unet(torch, nn, F):
    ConvBlock, _, _ = _make_common_classes(torch, nn, F)

    class PlainUNet(nn.Module):
        architecture_key = "unet"
        backbone_name = "none"
        input_channels = 1
        def __init__(self):
            super().__init__()
            self.encoder1 = ConvBlock(1, 48)
            self.encoder2 = ConvBlock(48, 96)
            self.encoder3 = ConvBlock(96, 192)
            self.encoder4 = ConvBlock(192, 320)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(320, 512)
            self.up4 = nn.ConvTranspose2d(512, 320, 2, 2)
            self.dec4 = ConvBlock(640, 320)
            self.up3 = nn.ConvTranspose2d(320, 192, 2, 2)
            self.dec3 = ConvBlock(384, 192)
            self.up2 = nn.ConvTranspose2d(192, 96, 2, 2)
            self.dec2 = ConvBlock(192, 96)
            self.up1 = nn.ConvTranspose2d(96, 48, 2, 2)
            self.dec1 = ConvBlock(96, 48)
            self.head = nn.Conv2d(48, 1, 1)
        def forward(self, x):
            e1 = self.encoder1(x)
            e2 = self.encoder2(self.pool(e1))
            e3 = self.encoder3(self.pool(e2))
            e4 = self.encoder4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)
        def encoder_parameters(self):
            for module in (self.encoder1, self.encoder2, self.encoder3, self.encoder4):
                yield from module.parameters()
        def decoder_parameters(self):
            encoder_ids = {id(p) for p in self.encoder_parameters()}
            return (p for p in self.parameters() if id(p) not in encoder_ids)
    return PlainUNet()


def _build_dino_family(architecture_key, encoder, torch, nn, F):
    ConvBlock, UpFuse, ResidualFusion = _make_common_classes(torch, nn, F)
    embed_dim = int(getattr(encoder, "embed_dim", getattr(encoder, "num_features", 384)))
    indices = _feature_indices(encoder)

    class DinoBase(nn.Module):
        input_channels = 1
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.feature_indices = indices
            self.embed_dim = embed_dim
        def extract(self, x):
            # DINOv2 accepts explicit intermediate block indices.
            feats = self.encoder.get_intermediate_layers(
                x,
                n=self.feature_indices,
                reshape=True,
                return_class_token=False,
            )
            if len(feats) != 4:
                raise RuntimeError(f"Expected 4 DINO feature maps, got {len(feats)}")
            return list(feats)
        def encoder_parameters(self):
            return self.encoder.parameters()
        def decoder_parameters(self):
            encoder_ids = {id(p) for p in self.encoder.parameters()}
            return (p for p in self.parameters() if id(p) not in encoder_ids)

    class DinoUNet(DinoBase):
        architecture_key = "dinov2_unet"
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleList([nn.Conv2d(embed_dim, c, 1) for c in (96, 128, 192, 256)])
            self.dec3 = UpFuse(256, 192, 192)
            self.dec2 = UpFuse(192, 128, 128)
            self.dec1 = UpFuse(128, 96, 96)
            self.refine = nn.Sequential(ConvBlock(96, 64), ConvBlock(64, 48))
            self.head = nn.Conv2d(48, 1, 1)
        def forward(self, x):
            h, w = x.shape[-2:]
            f1, f2, f3, f4 = self.extract(x)
            # Make transformer-depth features act as progressively coarser levels.
            targets = [
                (max(1, h // 4), max(1, w // 4)),
                (max(1, h // 8), max(1, w // 8)),
                (max(1, h // 16), max(1, w // 16)),
                (max(1, h // 32), max(1, w // 32)),
            ]
            fs = []
            for feat, proj, size in zip((f1, f2, f3, f4), self.proj, targets):
                fs.append(F.interpolate(proj(feat), size=size, mode="bilinear", align_corners=False))
            p1, p2, p3, p4 = fs
            y = self.dec3(p4, p3)
            y = self.dec2(y, p2)
            y = self.dec1(y, p1)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            return self.head(self.refine(y))

    class DinoDPT(DinoBase):
        architecture_key = "dinov2_dpt"
        def __init__(self, high_resolution=False):
            super().__init__()
            self.high_resolution = bool(high_resolution)
            ch = 192
            self.reassemble = nn.ModuleList([
                nn.Sequential(nn.Conv2d(embed_dim, ch, 1), _group_norm(nn, ch), nn.GELU())
                for _ in range(4)
            ])
            self.fuse4 = ResidualFusion(ch)
            self.fuse3 = ResidualFusion(ch)
            self.fuse2 = ResidualFusion(ch)
            self.fuse1 = ResidualFusion(ch)
            if self.high_resolution:
                self.hr1 = ConvBlock(1, 48)
                self.hr2 = ConvBlock(48, 72)
                self.hr3 = ConvBlock(72, 96)
                self.hr_down1 = nn.Conv2d(48, 48, 3, 2, 1)
                self.hr_down2 = nn.Conv2d(72, 72, 3, 2, 1)
                self.hr_fuse3 = UpFuse(ch, 96, 128)
                self.hr_fuse2 = UpFuse(128, 72, 96)
                self.hr_fuse1 = UpFuse(96, 48, 64)
                self.refine = ConvBlock(64, 48)
            else:
                self.refine = nn.Sequential(ConvBlock(ch, 96), ConvBlock(96, 48))
            self.head = nn.Conv2d(48, 1, 1)
        def forward(self, x):
            h, w = x.shape[-2:]
            f1, f2, f3, f4 = self.extract(x)
            targets = [
                (max(1, math.ceil(h / 4)), max(1, math.ceil(w / 4))),
                (max(1, math.ceil(h / 8)), max(1, math.ceil(w / 8))),
                (max(1, math.ceil(h / 16)), max(1, math.ceil(w / 16))),
                (max(1, math.ceil(h / 32)), max(1, math.ceil(w / 32))),
            ]
            levels = []
            for feat, proj, size in zip((f1, f2, f3, f4), self.reassemble, targets):
                levels.append(F.interpolate(proj(feat), size=size, mode="bilinear", align_corners=False))
            p1, p2, p3, p4 = levels
            y4 = self.fuse4(p4)
            y3 = self.fuse3(p3, y4)
            y2 = self.fuse2(p2, y3)
            y1 = self.fuse1(p1, y2)
            if self.high_resolution:
                s1 = self.hr1(x)
                s2 = self.hr2(self.hr_down1(s1))
                s3 = self.hr3(self.hr_down2(s2))
                y = self.hr_fuse3(y1, s3)
                y = self.hr_fuse2(y, s2)
                y = self.hr_fuse1(y, s1)
                y = self.refine(y)
            else:
                y = F.interpolate(y1, size=(h, w), mode="bilinear", align_corners=False)
                y = self.refine(y)
            return self.head(y)

    class PyramidPooling(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            bins = (1, 2, 3, 6)
            branch_ch = max(32, out_ch // len(bins))
            self.bins = bins
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(b),
                    nn.Conv2d(in_ch, branch_ch, 1, bias=False),
                    _group_norm(nn, branch_ch),
                    nn.GELU(),
                ) for b in bins
            ])
            self.bottleneck = ConvBlock(in_ch + branch_ch * len(bins), out_ch)
        def forward(self, x):
            h, w = x.shape[-2:]
            pooled = [x]
            for branch in self.branches:
                p = branch(x)
                p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
                pooled.append(p)
            return self.bottleneck(torch.cat(pooled, dim=1))

    class DinoUPerNet(DinoBase):
        architecture_key = "dinov2_upernet"
        def __init__(self):
            super().__init__()
            ch = 160
            self.lateral = nn.ModuleList([nn.Conv2d(embed_dim, ch, 1) for _ in range(4)])
            self.ppm = PyramidPooling(ch, ch)
            self.fpn_refine = nn.ModuleList([ConvBlock(ch, ch) for _ in range(3)])
            self.fuse = ConvBlock(ch * 4, 192)
            self.refine = nn.Sequential(ConvBlock(192, 96), ConvBlock(96, 48))
            self.head = nn.Conv2d(48, 1, 1)
        def forward(self, x):
            h, w = x.shape[-2:]
            feats = self.extract(x)
            targets = [
                (max(1, math.ceil(h / 4)), max(1, math.ceil(w / 4))),
                (max(1, math.ceil(h / 8)), max(1, math.ceil(w / 8))),
                (max(1, math.ceil(h / 16)), max(1, math.ceil(w / 16))),
                (max(1, math.ceil(h / 32)), max(1, math.ceil(w / 32))),
            ]
            lat = [
                F.interpolate(layer(feat), size=size, mode="bilinear", align_corners=False)
                for layer, feat, size in zip(self.lateral, feats, targets)
            ]
            lat[3] = self.ppm(lat[3])
            for i in (2, 1, 0):
                up = F.interpolate(lat[i + 1], size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
                lat[i] = self.fpn_refine[i](lat[i] + up)
            size0 = lat[0].shape[-2:]
            multi = [lat[0]] + [F.interpolate(v, size=size0, mode="bilinear", align_corners=False) for v in lat[1:]]
            y = self.fuse(torch.cat(multi, dim=1))
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            return self.head(self.refine(y))

    class DinoQueryMask(DinoBase):
        """Pure-PyTorch query-mask decoder inspired by Mask2Former.

        This is deliberately named QueryMask in LinkNex because it does not
        claim to reproduce the official Detectron2 Mask2Former implementation.
        It keeps the useful idea: pixel features + learned mask queries +
        transformer decoding, while remaining Windows/GUI friendly.
        """
        architecture_key = "dinov2_querymask"
        def __init__(self):
            super().__init__()
            hidden = 192
            self.pixel_proj = nn.ModuleList([nn.Conv2d(embed_dim, hidden, 1) for _ in range(4)])
            self.pixel_fuse = ConvBlock(hidden * 4, hidden)
            layer = nn.TransformerDecoderLayer(
                d_model=hidden,
                nhead=8,
                dim_feedforward=768,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=4)
            self.num_queries = 16
            self.query_embed = nn.Embedding(self.num_queries, hidden)
            self.query_class = nn.Linear(hidden, 1)
            self.query_mask = nn.Linear(hidden, hidden)
            self.pixel_refine = nn.Sequential(ConvBlock(hidden, 96), ConvBlock(96, 48))
            self.final_refine = ConvBlock(49, 48)
            self.head = nn.Conv2d(48, 1, 1)
        def forward(self, x):
            h, w = x.shape[-2:]
            feats = self.extract(x)
            # Fuse all transformer depths on the patch grid.
            base_size = feats[-1].shape[-2:]
            projected = [
                F.interpolate(proj(feat), size=base_size, mode="bilinear", align_corners=False)
                for proj, feat in zip(self.pixel_proj, feats)
            ]
            pixel = self.pixel_fuse(torch.cat(projected, dim=1))
            memory = pixel.flatten(2).transpose(1, 2)
            b = x.shape[0]
            queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
            q = self.transformer_decoder(queries, memory)
            class_logits = self.query_class(q).squeeze(-1)  # B,Q
            mask_embed = self.query_mask(q)                  # B,Q,C
            mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, pixel)
            # Differentiable binary semantic aggregation.
            query_score = torch.sigmoid(class_logits).unsqueeze(-1).unsqueeze(-1)
            mask_prob = torch.sigmoid(mask_logits)
            weighted = query_score * mask_prob
            semantic_prob = 1.0 - torch.prod(1.0 - weighted.clamp(0, 0.999), dim=1, keepdim=True)
            semantic_prob = semantic_prob.clamp(1e-5, 1.0 - 1e-5)
            semantic_logit = torch.logit(semantic_prob)
            semantic_logit = F.interpolate(semantic_logit, size=(h, w), mode="bilinear", align_corners=False)
            local = F.interpolate(pixel, size=(h, w), mode="bilinear", align_corners=False)
            local = self.pixel_refine(local)
            y = self.final_refine(torch.cat([local, semantic_logit], dim=1))
            return self.head(y)

    if architecture_key == "dinov2_unet":
        model = DinoUNet()
    elif architecture_key == "dinov2_dpt":
        model = DinoDPT(high_resolution=False)
    elif architecture_key == "dinov2_dpt_hr":
        model = DinoDPT(high_resolution=True)
        model.architecture_key = "dinov2_dpt_hr"
    elif architecture_key == "dinov2_upernet":
        model = DinoUPerNet()
    elif architecture_key == "dinov2_querymask":
        model = DinoQueryMask()
    else:
        raise ValueError(f"Unsupported DINO architecture: {architecture_key}")
    return model


def _find_first_input_conv(module, nn):
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3:
            return name, child
    return None, None


def _set_submodule(root, path: str, module):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def _build_segformer_b2(torch, nn, F, hf_model_dir: Path):
    try:
        from transformers import SegformerForSemanticSegmentation
    except Exception as exc:
        raise RuntimeError(
            "SegFormer-B2 requires the optional 'transformers' package.\n"
            "Install: pip install transformers safetensors\n"
            "Then run tools/download_pretrained_models.py to cache the model."
        ) from exc

    hf_model_dir = Path(hf_model_dir)
    if not hf_model_dir.exists():
        raise FileNotFoundError(
            "Local SegFormer-B2 weights were not found.\n\n"
            f"Expected folder:\n{hf_model_dir}\n\n"
            "Run tools/download_pretrained_models.py and select/download SegFormer-B2."
        )

    base = SegformerForSemanticSegmentation.from_pretrained(
        str(hf_model_dir),
        num_labels=1,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

    # True 1-channel input: average the first RGB projection kernel.
    name, conv = _find_first_input_conv(base.segformer, nn)
    if conv is None:
        raise RuntimeError("Could not locate SegFormer RGB input convolution.")
    _set_submodule(base.segformer, name, _replace_conv_in_channels(conv, 1, torch, nn))

    class SegFormerBinary(nn.Module):
        architecture_key = "segformer_b2"
        backbone_name = "segformer_b2"
        input_channels = 1
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            h, w = x.shape[-2:]
            out = self.model(pixel_values=x, return_dict=True).logits
            return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        def encoder_parameters(self):
            return self.model.segformer.parameters()
        def decoder_parameters(self):
            return self.model.decode_head.parameters()
    return SegFormerBinary(base)


def build_semantic_model(
    architecture_key: str,
    torch,
    nn,
    F,
    *,
    repo_dir: Optional[Path] = None,
    backbone_name: str = "dinov2_vits14",
    pretrained_path: Optional[Path] = None,
    hf_segformer_dir: Optional[Path] = None,
):
    architecture_key = str(architecture_key).strip().lower()
    if architecture_key not in MODEL_SPECS:
        raise ValueError(f"Unknown semantic architecture: {architecture_key}")

    load_info = {"missing": [], "unexpected": []}

    if architecture_key == "unet":
        model = _build_plain_unet(torch, nn, F)
        model.backbone_name = "none"
        return model, load_info

    if architecture_key == "segformer_b2":
        model = _build_segformer_b2(torch, nn, F, Path(hf_segformer_dir))
        return model, load_info

    if repo_dir is None:
        raise ValueError("repo_dir is required for DINOv2 architectures.")

    encoder, missing, unexpected = _build_dino_encoder(
        Path(repo_dir),
        backbone_name,
        Path(pretrained_path) if pretrained_path is not None else None,
        torch,
        nn,
    )
    model = _build_dino_family(architecture_key, encoder, torch, nn, F)
    model.backbone_name = backbone_name
    load_info = {"missing": missing, "unexpected": unexpected}
    return model, load_info


def optimizer_parameter_groups(model, encoder_lr: float, decoder_lr: float, weight_decay: float = 1e-4):
    """Return non-overlapping AdamW parameter groups for any LinkNex model."""
    encoder_params = list(model.encoder_parameters()) if hasattr(model, "encoder_parameters") else []
    encoder_ids = {id(p) for p in encoder_params}
    decoder_params = list(model.decoder_parameters()) if hasattr(model, "decoder_parameters") else [
        p for p in model.parameters() if id(p) not in encoder_ids
    ]

    # Remove accidental duplication and frozen params.
    seen = set()
    enc = []
    for p in encoder_params:
        if p.requires_grad and id(p) not in seen:
            enc.append(p); seen.add(id(p))
    dec = []
    for p in decoder_params:
        if p.requires_grad and id(p) not in seen:
            dec.append(p); seen.add(id(p))
    # Include any remaining trainable parameter (e.g. wrapper heads).
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            dec.append(p); seen.add(id(p))

    groups = []
    if enc:
        groups.append({"params": enc, "lr": float(encoder_lr), "weight_decay": float(weight_decay), "name": "encoder"})
    if dec:
        groups.append({"params": dec, "lr": float(decoder_lr), "weight_decay": float(weight_decay), "name": "decoder"})
    return groups
