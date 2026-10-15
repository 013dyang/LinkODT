# LinkODT pretrained model files

The semantic GUI now supports multiple architectures. All LinkODT-native
semantic models use **one grayscale input channel**.

## Easiest download method

From the project root, double-click:

`DOWNLOAD_MODELS.bat`

or run:

```powershell
python tools/download_pretrained_models.py
```

The script shows a menu and saves every weight in the exact location expected
by LinkODT.

You can also download one model directly:

```powershell
python tools/download_pretrained_models.py --model dinov2_vits14
python tools/download_pretrained_models.py --model dinov2_vitb14
python tools/download_pretrained_models.py --model dinov2_vits14_reg
python tools/download_pretrained_models.py --model dinov2_vitb14_reg
python tools/download_pretrained_models.py --model all-dino
```

## DINOv2 weights

Expected files:

```text
models/
├─ dinov2_vits14_pretrain.pth
├─ dinov2_vitb14_pretrain.pth
├─ dinov2_vits14_reg4_pretrain.pth
└─ dinov2_vitb14_reg4_pretrain.pth
```

The downloader uses the official Meta/Facebook AI public DINOv2 weight URLs.
The same links are listed in the bundled `dinov2-main/README.md` and the
official DINOv2 repository.

Recommended use:

- ViT-S/14: faster development / lower VRAM
- ViT-B/14: recommended final public model
- register variants: optional architecture comparison

The original DINOv2 weights are RGB. LinkODT first loads the official weights,
then averages the three patch-embedding input kernels into a **true 1-channel**
patch embedding. A microscopy SSL encoder produced by the updated SSL script
is already one-channel and is loaded directly.

## SegFormer-B2 external baseline

First install the optional packages:

```powershell
pip install -r requirements_semantic_optional.txt
```

Then:

```powershell
python tools/download_pretrained_models.py --model segformer_b2
```

The model is cached here:

```text
models/hf_segformer_b2_ade/
```

Source model:

`nvidia/segformer-b2-finetuned-ade-512-512`

LinkODT adapts its first RGB projection convolution to one grayscale channel.
This is an external baseline, not the default LinkODT model.

## Which GUI models need which weights?

| GUI architecture | Extra pretrained asset |
|---|---|
| U-Net | none |
| DINOv2 + U-Net | selected DINOv2 backbone |
| DINOv2 + DPT | selected DINOv2 backbone |
| DINOv2 + DPT-HR | selected DINOv2 backbone or microscopy SSL encoder |
| DINOv2 + UPerNet | selected DINOv2 backbone |
| DINOv2 + QueryMask | selected DINOv2 backbone |
| SegFormer-B2 | local Hugging Face SegFormer-B2 folder |

## Microscopy SSL encoder

After running:

`ssl/train_dinov2_microscopy_ssl.py`

the encoder-only `.pth` file can be selected in the GUI under:

`Pretrained Encoder`.

Make sure the GUI Backbone selection matches the SSL backbone (ViT-S vs ViT-B,
and register vs non-register).

## Optional QC weight

For fully offline ResNet18 QC training, place either
`resnet18-f37072fd.pth` or `resnet18_imagenet.pth` in this folder.
