Markdown
# Project Setup Guide

This guide provides instructions for installing Blender, setting up environment variables, and running scripts to process `.glb` files, generate camera angles, and create depth, normal, and alpha channel files.

## Blender Installation

Follow these steps to install Blender 3.2.0 on Windows:

1. **Download Blender:**

```bash
wget [https://download.blender.org/release/Blender3.2/blender-3.2.0-windows-x64.zip](https://download.blender.org/release/Blender3.2/blender-3.2.0-windows-x64.zip)
Use code with caution.

Extract the downloaded file:

Bash
tar -xf blender-3.2.0-windows-x64.zip
Use code with caution.

Set the Environment Variable:

Add the Blender directory to your system's PATH variable to run Blender from the command line.

Running Scripts on Your GLB Folder
1. Rendering GLB Files
To render images from .glb files, run the following script:

Bash
python /objaverse-xl/scripts/rendering/blender_python.py --directory /path/to/glb/files
Use code with caution.

This script generates a separate folder for each .glb file, with images saved under the rendering_random_32views directory within each folder.

2. Generating Camera Angles
After rendering the images, generate camera angles using:

Bash
python /extras/npz_generator.py --root_directory /path/to/rendering_random_32views
Use code with caution.

This script creates camera angle files in each folder containing the rendered images.

3. Generating Depth and Normal Files
To generate depth and normal files, use:

Bash
python /objaverse-xl/GeoWizard/geowizard/run_infer.py --input_dir "/path/to/root/directory" --pretrained_model_path "lemonaddie/geowizard" --domain "object"
Use code with caution.

This will produce .depth and .normal files in the specified directory.

4. Including Alpha Channels
To add alpha channels to your images, execute:

Bash
python /extras/making_alpha_chanels.py --input_dir /path/to/your/directory
Use code with caution.

This script processes images in the specified directory to include alpha channels.
