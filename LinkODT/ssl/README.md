# Microscopy DINOv2 self-supervised adaptation

The SSL script is the domain-adaptation stage before supervised segmentation.
It now trains on **true single-channel grayscale microscopy input**.

## Recommended workflow

```text
Official DINOv2 pretrained encoder
        ↓
Microscopy DINO self-supervised adaptation
        ↓
encoder-only microscopy checkpoint
        ↓
LinkNex GUI: DINOv2 + DPT-HR (or another DINO decoder)
        ↓
supervised image/mask segmentation
```

## 1. Download the official initialization

From the project root:

```powershell
python tools/download_pretrained_models.py --model dinov2_vitb14
```

For faster development use `dinov2_vits14` instead.

## 2. Configure the SSL script

Open:

`ssl/train_dinov2_microscopy_ssl.py`

At the USER CONFIG section set at least:

```python
DATA_DIR = r"D:\your\unlabelled_microscopy_images"
OUTPUT_DIR = r"D:\your\ssl_output"
BACKBONE_NAME = "dinov2_vitb14"  # or dinov2_vits14
```

The script automatically chooses the matching official file under `models/`.
Supported backbones:

- `dinov2_vits14`
- `dinov2_vitb14`
- `dinov2_vits14_reg`
- `dinov2_vitb14_reg`

## 3. Run

```powershell
python ssl/train_dinov2_microscopy_ssl.py
```

The script loads the official RGB DINOv2 checkpoint first and averages its RGB
patch-embedding kernels into one channel. Subsequent SSL training and saved
microscopy encoders are therefore **1-channel**.

It saves resumable SSL checkpoints and encoder-only checkpoints under the
configured output directory. The encoder-only file is the one to browse from
the LinkNex semantic training GUI.

## 4. Supervised segmentation

In LinkNex:

1. choose `Pretrained encoder · New segmentation head`;
2. choose the same DINOv2 Backbone used during SSL;
3. choose a segmentation architecture (recommended: `DINOv2 + DPT-HR`);
4. browse the microscopy SSL encoder `.pth`;
5. choose the image/mask dataset and Train.

The same microscopy encoder can be benchmarked with DPT, DPT-HR, U-Net,
UPerNet and QueryMask heads, provided the selected DINO backbone is identical.
