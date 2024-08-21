import os
import numpy as np
import argparse

def save_npz_for_each_folder(root_dir):
    # Walk through the directory
    for subdir, dirs, files in os.walk(root_dir):
        # Filter out only the npy files
        npy_files = [file for file in files if file.endswith('.npy')]
        
        if npy_files:
            # Create a list to hold the npy data
            cam_poses = []

            for npy_file in npy_files:
                # Load the npy file
                data = np.load(os.path.join(subdir, npy_file))
                # Store it in the list
                cam_poses.append(data)
            
            # Determine the output npz file path
            npz_file_name = "cameras.npz"
            npz_file_path = os.path.join(subdir, npz_file_name)
            
            # Save the list to the npz file
            np.savez(npz_file_path, cam_poses=cam_poses)
            print(f"Saved: {npz_file_path}")

# Set up the argument parser
parser = argparse.ArgumentParser(description="Save npz files for each folder containing .npy files")
parser.add_argument(
    "--root_directory", 
    type=str, 
    required=True, 
    help="The root directory to walk through and process .npy files"
)

args = parser.parse_args()

# Replace this with the actual path to your root directory
root_directory = args.root_directory
save_npz_for_each_folder(root_directory)
