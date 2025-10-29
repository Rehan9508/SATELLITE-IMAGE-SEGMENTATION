import argparse
from pathlib import Path
import cv2
import numpy as np


def generate_random_colors(num_colors: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(num_colors, 3), dtype=np.uint8)
    return colors


def colorize_mask(mask_gray: np.ndarray) -> np.ndarray:
    unique_classes = np.unique(mask_gray)
    colors = generate_random_colors(len(unique_classes))
    color_map = np.zeros((*mask_gray.shape, 3), dtype=np.uint8)
    for i, cls in enumerate(unique_classes):
        color_map[mask_gray == cls] = colors[i]
    return color_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Colorize a grayscale segmentation mask and save PNG output")
    parser.add_argument("--image", type=str, required=True, help="Path to the input RGB image (for reference only)")
    parser.add_argument("--mask", type=str, required=True, help="Path to the grayscale mask image")
    parser.add_argument("--output", type=str, default="segmented_output.png", help="Output PNG path")
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)

    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    if not mask_path.is_file():
        raise SystemExit(f"Mask not found: {mask_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise SystemExit(f"Failed to load image: {image_path}")
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise SystemExit(f"Failed to load mask: {mask_path}")

    color_map = colorize_mask(mask_gray)
    # Save as BGR for OpenCV write
    color_map_bgr = cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, color_map_bgr)
    print(f"Saved colorized mask to {args.output}")


if __name__ == "__main__":
    main()


