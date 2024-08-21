import os
from transformers import pipeline
from PIL import Image
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Generate depth maps for .png files in a directory")
parser.add_argument(
    "--base_dir", 
    type=str, 
    required=True, 
    help="The base directory containing subdirectories with .png files"
)
parser.add_argument(
    "--device", 
    type=int, 
    default=0, 
    help="The GPU device number to use (default is 0)"
)

args = parser.parse_args()

# Load the pipeline and set it to use the specified GPU device
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=args.device)
print("Pipeline loaded successfully.")

# Base directory containing the subdirectories
base_dir = args.base_dir

# Traverse through all subdirectories
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.png'):  # Process only .png files
            # Construct the full path to the image file
            image_path = os.path.join(root, file)
            print(f"Processing {image_path}...")

            # Load the image
            image = Image.open(image_path)

            # Inference
            depth = pipe(image)["depth"]

            # Save the depth map with the new filename
            depth_filename = os.path.splitext(file)[0] + '_depth.png'
            output_path = os.path.join(root, depth_filename)
            depth.save(output_path)

            print(f"Depth map saved to {output_path}")

print("All depth maps have been generated.")
