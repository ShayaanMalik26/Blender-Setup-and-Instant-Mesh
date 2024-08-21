import argparse
import os
from PIL import Image
import numpy
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from scipy.ndimage import binary_erosion
from typing import Tuple

# Function to compute alpha and process images
def postprocess(rgb_img: Image.Image, normal_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    normal_vecs_pred = numpy.array(normal_img, dtype=numpy.float64) / 255.0 * 2 - 1
    alpha_pred = numpy.linalg.norm(normal_vecs_pred, axis=-1)

    is_foreground = alpha_pred > 0.6
    is_background = alpha_pred < 0.2
    structure = numpy.ones((4, 4), dtype=numpy.uint8)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = numpy.full(alpha_pred.shape, dtype=numpy.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    # Ensure there's at least one background pixel
    if not numpy.any(is_background):
        trimap[alpha_pred == numpy.min(alpha_pred)] = 0  # Force the minimum alpha pixel to be background

    # Convert image to RGB before processing
    img_normalized = numpy.array(rgb_img.convert("RGB"), dtype=numpy.float64) / 255.0
    trimap_normalized = trimap.astype(numpy.float64) / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = numpy.clip(cutout * 255, 0, 255).astype(numpy.uint8)
    cutout = Image.fromarray(cutout)

    normal_vecs_pred = normal_vecs_pred / (numpy.linalg.norm(normal_vecs_pred, axis=-1, keepdims=True) + 1e-8)
    normal_vecs_pred = normal_vecs_pred * 0.5 + 0.5
    normal_vecs_pred = normal_vecs_pred * alpha[..., None] + 0.5 * (1 - alpha[..., None])
    normal_image_normalized = numpy.clip(normal_vecs_pred * 255, 0, 255).astype(numpy.uint8)

    return cutout, Image.fromarray(normal_image_normalized)

# Function to compute alpha and process images
def postprocess(rgb_img: Image.Image, normal_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
    normal_vecs_pred = np.array(normal_img, dtype=np.float64) / 255.0 * 2 - 1
    alpha_pred = np.linalg.norm(normal_vecs_pred, axis=-1)

    is_foreground = alpha_pred > 0.6
    is_background = alpha_pred < 0.2
    structure = np.ones((4, 4), dtype=np.uint8)

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(alpha_pred.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    if not np.any(is_background):
        trimap[alpha_pred == np.min(alpha_pred)] = 0

    img_normalized = np.array(rgb_img.convert("RGB"), dtype=np.float64) / 255.0
    trimap_normalized = trimap.astype(np.float64) / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    normal_vecs_pred = normal_vecs_pred / (np.linalg.norm(normal_vecs_pred, axis=-1, keepdims=True) + 1e-8)
    normal_vecs_pred = normal_vecs_pred * 0.5 + 0.5
    normal_vecs_pred = normal_vecs_pred * alpha[..., None] + 0.5 * (1 - alpha[..., None])
    normal_image_normalized = np.clip(normal_vecs_pred * 255, 0, 255).astype(np.uint8)

    return cutout, Image.fromarray(normal_image_normalized)

# Function to traverse directories and process images
def process_directory(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_normal.png") and not file.endswith("_depth.png") and not file.endswith("_depth_colored.png") and not file.endswith("_normal_colored.png"):
                base_name = file.replace("_normal.png", "")
                normal_path = os.path.join(subdir, file)
                original_path = os.path.join(subdir, f"{base_name}_original.png")
                output_path = os.path.join(subdir, f"{base_name}.png")

                if os.path.exists(normal_path):
                    os.rename(normal_path, original_path)
                    normal_img = Image.open(original_path)
                    rgb_img = Image.open(original_path)

                    # Process images
                    cutout, _ = postprocess(rgb_img, normal_img)

                    # Save output image
                    cutout.save(output_path)
                    print(f"Saved {output_path}")
                else:
                    print(f"Normal image not found for {base_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to compute alpha channels.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing the images.")
    
    args = parser.parse_args()
    
    process_directory(args.input_dir)