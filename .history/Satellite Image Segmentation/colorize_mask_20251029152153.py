import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_random_colors(num_colors: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(num_colors, 3), dtype=np.uint8)


def colorize_mask(mask_path: Path, output_path: Path) -> None:
    pred_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if pred_mask is None:
        raise FileNotFoundError(f"Could not read mask from {mask_path}")

    unique_classes = np.unique(pred_mask)
    colors = generate_random_colors(len(unique_classes))

    segmentation_map = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for i, class_id in enumerate(unique_classes):
        segmentation_map[pred_mask == class_id] = colors[i]

    cv2.imwrite(str(output_path), cv2.cvtColor(segmentation_map, cv2.COLOR_RGB2BGR))
    print(f"Saved colorized mask to {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Colorize a grayscale segmentation mask")
    p.add_argument("--mask", required=True, type=Path)
    p.add_argument("--out", required=False, type=Path, default=Path("segmented_output.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    colorize_mask(args.mask, args.out)


if __name__ == "__main__":
    main()


