
import os
import csv
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# 建议目录结构：
#
# project/
# ├─ evaluate_masks_with_panels.py
# ├─ original_images/
# ├─ prediction/
# └─ ground_truth/
#
# original_images : 原始图像，例如 cell001.tif
# prediction      : 预测 mask，例如 cell001_mask.tif
# ground_truth    : 真实 mask，例如 cell001_mask.tif

BASE_DIR = Path("./data/predict_score")

IMAGE_DIR = BASE_DIR / "original_images"
PRED_DIR = BASE_DIR / "prediction"
GT_DIR = BASE_DIR / "ground_truth"

OUTPUT_PER_IMAGE = BASE_DIR / "mask_evaluation_per_image.csv"
OUTPUT_SUMMARY = BASE_DIR / "mask_evaluation_summary.csv"

PANEL_DIR = BASE_DIR / "comparison_panels"
PANEL_DIR.mkdir(exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".tif", ".tiff",
    ".png", ".jpg", ".jpeg",
    ".bmp"
}

# ============================================================
# Utility
# ============================================================

def normalize_stem(filename: str) -> str:
    """
    归一化文件名 stem，用于图像、预测、GT 的自动配对。
    """
    stem = Path(filename).stem.lower()

    suffixes = [
        "_prediction",
        "_predicted",
        "_predict",
        "_pred",
        "_mask",
        "_gt",
        "_ground_truth",
        "_groundtruth",
        "_img",
        "_image",
        "_raw",
    ]

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                changed = True

    return stem


def collect_files(folder: Path):
    files = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def read_image(path: Path) -> np.ndarray:
    """
    读取 2D / 3D 图像。
    TIFF 用 tifffile，其余用 imageio。
    """
    ext = path.suffix.lower()

    if ext in {".tif", ".tiff"}:
        img = tifffile.imread(path)
    else:
        import imageio.v3 as iio
        img = iio.imread(path)

    img = np.asarray(img)

    # RGB/RGBA -> 取第一通道（如果是普通彩色图）
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        # 这种一般是 2D RGB 图
        if img.shape[0] > 4 and img.shape[1] > 4:
            img = img[..., 0]

    return img


def to_binary(mask: np.ndarray) -> np.ndarray:
    """
    0/1、0/255、任意 >0 均视为 foreground。
    """
    return np.asarray(mask) > 0


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: prediction={pred.shape}, GT={gt.shape}"
        )

    pred_flat = pred.ravel()
    gt_flat = gt.ravel()

    tp = np.logical_and(pred_flat, gt_flat).sum()
    tn = np.logical_and(~pred_flat, ~gt_flat).sum()
    fp = np.logical_and(pred_flat, ~gt_flat).sum()
    fn = np.logical_and(~pred_flat, gt_flat).sum()

    total = tp + tn + fp + fn

    dice_den = 2 * tp + fp + fn
    dice = 1.0 if dice_den == 0 else (2 * tp) / dice_den

    iou_den = tp + fp + fn
    iou = 1.0 if iou_den == 0 else tp / iou_den

    precision_den = tp + fp
    precision = (
        1.0 if precision_den == 0 and (tp + fn) == 0
        else (0.0 if precision_den == 0 else tp / precision_den)
    )

    recall_den = tp + fn
    recall = (
        1.0 if recall_den == 0 and (tp + fp) == 0
        else (0.0 if recall_den == 0 else tp / recall_den)
    )

    specificity_den = tn + fp
    specificity = np.nan if specificity_den == 0 else tn / specificity_den

    accuracy = np.nan if total == 0 else (tp + tn) / total

    pred_foreground = int(pred_flat.sum())
    gt_foreground = int(gt_flat.sum())

    if gt_foreground == 0:
        foreground_error_percent = 0.0 if pred_foreground == 0 else np.nan
    else:
        foreground_error_percent = (
            (pred_foreground - gt_foreground) / gt_foreground * 100.0
        )

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "Specificity": float(specificity),
        "Accuracy": float(accuracy),
        "Pred_foreground": pred_foreground,
        "GT_foreground": gt_foreground,
        "Foreground_error_percent": float(foreground_error_percent),
    }


def get_middle_slice(arr: np.ndarray) -> np.ndarray:
    """
    对 2D / 3D 数据取显示用的中间层。
    - 2D: 直接返回
    - 3D: 返回 Z 中间层
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        z_mid = arr.shape[0] // 2
        return arr[z_mid]

    raise ValueError(f"Unsupported ndim for display: {arr.ndim}, shape={arr.shape}")


def normalize_for_display(img2d: np.ndarray) -> np.ndarray:
    """
    原图显示归一化到 0~1。
    """
    img2d = np.asarray(img2d, dtype=np.float32)

    finite = np.isfinite(img2d)
    if not np.any(finite):
        return np.zeros_like(img2d, dtype=np.float32)

    vals = img2d[finite]
    vmin = np.min(vals)
    vmax = np.max(vals)

    if vmax == vmin:
        return np.zeros_like(img2d, dtype=np.float32)

    out = (img2d - vmin) / (vmax - vmin)
    out = np.clip(out, 0, 1)
    return out


def save_comparison_panel(
    image_raw: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    out_path: Path,
    dice_value: float,
    title_stem: str
):
    """
    保存三联图：
    原图 | 预测 mask | GT mask
    右下角写 Dice=xx.xx
    对 3D 只显示中间层。
    """
    image2d = normalize_for_display(get_middle_slice(image_raw))
    pred2d = get_middle_slice(pred_mask).astype(np.uint8)
    gt2d = get_middle_slice(gt_mask).astype(np.uint8)

    fig = plt.figure(figsize=(12, 4))
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]

    axes[0].imshow(image2d, cmap="gray")
    axes[0].set_title("Original")

    axes[1].imshow(pred2d, cmap="gray")
    axes[1].set_title("Prediction")

    axes[2].imshow(gt2d, cmap="gray")
    axes[2].set_title("Ground Truth")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title_stem, fontsize=12)

    # 右下角显示 Dice，保留两位小数
    fig.text(
        0.98, 0.04,
        f"Dice={dice_value:.2f}",
        ha="right",
        va="bottom",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Mask evaluation with comparison panels")
    print("=" * 70)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Original image folder not found:\n{IMAGE_DIR}"
        )

    if not PRED_DIR.exists():
        raise FileNotFoundError(
            f"Prediction folder not found:\n{PRED_DIR}"
        )

    if not GT_DIR.exists():
        raise FileNotFoundError(
            f"Ground truth folder not found:\n{GT_DIR}"
        )

    image_files = collect_files(IMAGE_DIR)
    pred_files = collect_files(PRED_DIR)
    gt_files = collect_files(GT_DIR)

    print(f"Original images   : {len(image_files)}")
    print(f"Prediction masks  : {len(pred_files)}")
    print(f"Ground truth masks: {len(gt_files)}")

    # --------------------------------------------------------
    # 所有文件统一用 normalize_stem() 配对
    #
    # 例如：
    # original_images/cell001.tif
    # prediction/cell001_mask.tif
    # ground_truth/cell001_mask.tif
    #
    # 三者都会归一化为：
    # cell001
    # --------------------------------------------------------
    image_lookup = {normalize_stem(p.name): p for p in image_files}
    pred_lookup = {normalize_stem(p.name): p for p in pred_files}
    gt_lookup = {normalize_stem(p.name): p for p in gt_files}

    results = []
    unmatched_predictions = []

    for index, pred_path in enumerate(pred_files, start=1):

        key = normalize_stem(pred_path.name)

        # 原图：cell001.tif
        if key not in image_lookup:
            print(f"[NO ORIGINAL IMAGE] {pred_path.name}")
            unmatched_predictions.append(pred_path.name)
            continue

        # GT：cell001_mask.tif / cell001_gt.tif / cell001.tif 都可以
        if key not in gt_lookup:
            print(f"[NO GT] {pred_path.name}")
            unmatched_predictions.append(pred_path.name)
            continue

        image_path = image_lookup[key]
        gt_path = gt_lookup[key]

        print(
            f"[{index}/{len(pred_files)}] "
            f"{pred_path.name}  <->  {gt_path.name}  <->  {image_path.name}"
        )

        try:
            image_raw = read_image(image_path)
            pred_raw = read_image(pred_path)
            gt_raw = read_image(gt_path)

            pred = to_binary(pred_raw)
            gt = to_binary(gt_raw)

            if pred.shape != gt.shape:
                raise ValueError(
                    f"Prediction and GT shape mismatch: "
                    f"pred={pred.shape}, gt={gt.shape}"
                )

            # 原图与 mask 尺寸也需要一致（或至少 XY/Z 兼容）
            if image_raw.shape != pred.shape:
                raise ValueError(
                    f"Original image and mask shape mismatch: "
                    f"image={image_raw.shape}, mask={pred.shape}"
                )

            metrics = calculate_metrics(pred, gt)

            row = {
                "image_file": image_path.name,
                "prediction_file": pred_path.name,
                "ground_truth_file": gt_path.name,
                "shape": str(pred.shape),
                **metrics
            }
            results.append(row)

            panel_path = PANEL_DIR / f"{Path(pred_path).stem}_panel.png"
            save_comparison_panel(
                image_raw=image_raw,
                pred_mask=pred,
                gt_mask=gt,
                out_path=panel_path,
                dice_value=metrics["Dice"],
                title_stem=pred_path.stem
            )

            print(
                f"    Dice={metrics['Dice']:.4f} | "
                f"IoU={metrics['IoU']:.4f} | "
                f"Panel saved: {panel_path.name}"
            )

        except Exception as e:
            print(f"    [ERROR] {e}")

            results.append({
                "image_file": image_path.name if 'image_path' in locals() else "",
                "prediction_file": pred_path.name,
                "ground_truth_file": gt_path.name if 'gt_path' in locals() else "",
                "shape": "",
                "TP": "",
                "TN": "",
                "FP": "",
                "FN": "",
                "Dice": "",
                "IoU": "",
                "Precision": "",
                "Recall": "",
                "Specificity": "",
                "Accuracy": "",
                "Pred_foreground": "",
                "GT_foreground": "",
                "Foreground_error_percent": "",
                "error": str(e),
            })

    if not results:
        print("\nNo paired masks were evaluated.")
        return

    # ========================================================
    # Save per-image CSV
    # ========================================================
    fieldnames = [
        "image_file",
        "prediction_file",
        "ground_truth_file",
        "shape",
        "TP",
        "TN",
        "FP",
        "FN",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "Specificity",
        "Accuracy",
        "Pred_foreground",
        "GT_foreground",
        "Foreground_error_percent",
        "error",
    ]

    with open(OUTPUT_PER_IMAGE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            if "error" not in row:
                row["error"] = ""
            writer.writerow(row)

    # ========================================================
    # Save summary CSV
    # ========================================================
    valid_results = [
        r for r in results
        if isinstance(r.get("Dice"), (int, float, np.floating))
    ]

    metric_names = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "Specificity",
        "Accuracy",
        "Foreground_error_percent",
    ]

    summary_rows = []
    for metric_name in metric_names:
        values = np.array(
            [r[metric_name] for r in valid_results if np.isfinite(r[metric_name])],
            dtype=float
        )

        if len(values) == 0:
            continue

        summary_rows.append({
            "metric": metric_name,
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })

    with open(OUTPUT_SUMMARY, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "n", "mean", "std", "median", "min", "max"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # ========================================================
    # Print final summary
    # ========================================================
    print("\n" + "=" * 70)
    print("Evaluation completed")
    print("=" * 70)

    print(f"Paired masks evaluated: {len(valid_results)}")
    print(f"Unmatched predictions  : {len(unmatched_predictions)}")

    if valid_results:
        dice_values = np.array([r["Dice"] for r in valid_results], dtype=float)
        iou_values = np.array([r["IoU"] for r in valid_results], dtype=float)
        print(f"\nMean Dice : {np.mean(dice_values):.4f}")
        print(f"Mean IoU  : {np.mean(iou_values):.4f}")

    print("\nOutput files:")
    print(OUTPUT_PER_IMAGE)
    print(OUTPUT_SUMMARY)
    print(PANEL_DIR)


if __name__ == "__main__":
    main()

