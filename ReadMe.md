

Here's the complete README.md file in markdown format:

markdown
Copy code
# Project Setup Guide

This guide provides instructions for installing Blender, setting up environment variables, and running scripts to process `.glb` files, generate camera angles, and create depth, normal, and alpha channel files.

## Blender Installation

Follow these steps to install Blender 3.2.0 on Windows:

1. **Download Blender:**
   ```bash
   wget https://download.blender.org/release/Blender3.2/blender-3.2.0-windows-x64.zip
Extract the downloaded file:

bash
Copy code
tar -xf blender-3.2.0-windows-x64.zip
Set the Environment Variable:
Add the Blender directory to your system's PATH variable to run Blender from the command line.

Running Scripts on Your GLB Folder
## 1. Rendering GLB Files
To render images from .glb files, run the following script:

bash
Copy code
python /objaverse-xl/scripts/rendering/blender_python.py --directory /path/to/glb/files
This script generates a separate folder for each .glb file, with images saved under the rendering_random_32views directory within each folder.

### Structure 

'''--input-dir
---rendering_random_32views
------rendered_image_folder
-----------------rendered images(png and npy)'''


## 2. Generating Camera Angles
After rendering the images, generate camera angles using:

bash
Copy code
python /extras/npz_generator.py --root_directory /path/to/rendering_random_32views
This script creates camera angle files in each folder containing the rendered images.

## 3. Generating Depth and Normal Files
To generate depth and normal files, use:

bash
python /objaverse-xl/GeoWizard/geowizard/run_infer.py --input_dir "/path/to/root/directory" --pretrained_model_path "lemonaddie/geowizard" --domain "object"
This will produce .depth and .normal files in the specified directory.

## 4. Including Alpha Channels
To add alpha channels to your images, execute:

bash

python /extras/making_alpha_chanels.py --input_dir /path/to/your/directory
This script processes images in the specified directory to include alpha channels.


## Clone the InstantMesh Repo 
https://github.com/TencentARC/InstantMesh.git

And swap the files upload on the repo which are modified.

Additional Notes
Ensure all dependencies are installed, and the environment variables are properly configured before running the scripts.
Customize the paths in the commands to match your project structure and file locations.
By following these steps, you can efficiently set up your environment, process .glb files, and generate the necessary outputs for your project.



