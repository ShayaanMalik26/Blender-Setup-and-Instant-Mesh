import torch
from safetensors import safe_open
from safetensors.torch import save_file

def remove_extra_keys_and_clean_prefix(finetuned_path, pretrained_path, output_path):
    # Load pretrained model keys
    with safe_open(pretrained_path, framework="pt", device="cpu") as f:
        pretrained_keys = set(f.keys())
    
    # Load fine-tuned model
    with safe_open(finetuned_path, framework="pt", device="cpu") as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    
    cleaned_tensors = {}
    
    for key in list(tensors.keys()):
        # Remove unet.unet. prefix
        clean_key = key.replace("unet.unet.", "")

        # Only keep the key if it is in the pretrained keys
        if clean_key in pretrained_keys:
            cleaned_tensors[clean_key] = tensors[key]

    # Save the cleaned tensors to a new safetensors file
    save_file(cleaned_tensors, output_path)

# Example usage
finetuned_path = "/fsx/ubuntu/3d_model_finetuning/InstantMesh/ckpts/models--sudo-ai--zero123plus-v1.2/snapshots/2da07e89919e1a130c9b5add1584c70c7aa065fd/unet/diffusion_pytorch_model.safetensors"
pretrained_path = "/fsx/ubuntu/3d_model_finetuning/InstantMesh/ckpts/models--sudo-ai--zero123plus-v1.2/snapshots/2da07e89919e1a130c9b5add1584c70c7aa065fd/unet/pretrained_diffusion_pytorch_model.safetensors"
output_path = "/fsx/ubuntu/3d_model_finetuning/InstantMesh/ckpts/models--sudo-ai--zero123plus-v1.2/snapshots/2da07e89919e1a130c9b5add1584c70c7aa065fd/unet/cleaned_diffusion_pytorch_model.safetensors"

remove_extra_keys_and_clean_prefix(finetuned_path, pretrained_path, output_path)
