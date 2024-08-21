import os
import torch
from safetensors import safe_open
from pprint import pprint

def get_all_safetensors_keys(root_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.safetensors'):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing file: {file_path}")

                tensors = {}
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)

                parent_folder = os.path.basename(dirpath)
                output_filename = f"{parent_folder}_{os.path.splitext(filename)[0]}.txt"
                output_path = os.path.join(output_dir, output_filename)

                with open(output_path, 'w') as outfile:
                    outfile.write("\n".join(tensors.keys()))

                pprint(f"Keys for {filename} saved to {output_path}")

# Example usage:
root_path = "/fsx/ubuntu/3d_model_finetuning/InstantMesh/ckpts/models--sudo-ai--zero123plus-v1.2/snapshots"
output_dir = "/fsx/ubuntu/3d_model_finetuning/InstantMesh/keys_output"

get_all_safetensors_keys(root_path, output_dir)
