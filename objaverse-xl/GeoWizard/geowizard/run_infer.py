import os
import logging
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
import argparse

from models.geowizard_pipeline import DepthNormalEstimationPipeline
from utils.seed_all import seed_all

from diffusers import DDIMScheduler, AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run MonoDepthNormal Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='lemonaddie/geowizard',
        help="Pretrained model path from Hugging Face or local directory",
    )    
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input root directory containing subdirectories with PNG files."
    )

    parser.add_argument(
        "--output_dir", type=str, required=False, help="Output directory (not used)."
    )
    parser.add_argument(
        "--domain",
        type=str,
        default='indoor',
        required=True,
        help="Domain prediction",
    )   

    # Inference settings
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=10,
        help="Diffusion denoising steps; more steps result in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled; more inferences give better results but run slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal results.",
    )

    # Resolution settings
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, output depth at resized operating resolution. Default: False.",
    )

    # Depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    # Other settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    
    args = parser.parse_args()
    
    checkpoint_path = args.pretrained_model_path
    input_dir = args.input_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size > 15:
        logging.warning("Long ensemble steps, low speed..")
    
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    domain = args.domain

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    
    if batch_size == 0:
        batch_size = 1  # set default batchsize
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # Declare a pipeline
    vae = AutoencoderKL.from_pretrained(checkpoint_path, subfolder='vae')
    scheduler = DDIMScheduler.from_pretrained(checkpoint_path, subfolder='scheduler')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(checkpoint_path, subfolder="image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(checkpoint_path, subfolder="feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet")

    pipe = DepthNormalEstimationPipeline(vae=vae,
                                image_encoder=image_encoder,
                                feature_extractor=feature_extractor,
                                unet=unet,
                                scheduler=scheduler)

    logging.info("Loading pipeline successfully.")

    seed_all(seed)
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # Run without xformers

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for root, dirs, files in os.walk(input_dir):
            for file in tqdm(files, desc="Estimating Depth & Normal", leave=True):
                if file.endswith('.png'):
                    rgb_path = os.path.join(root, file)

                    # Read input image
                    input_image = Image.open(rgb_path)

                    # Predict the depth & normal
                    pipe_out = pipe(input_image,
                                    denoising_steps=denoise_steps,
                                    ensemble_size=ensemble_size,
                                    processing_res=processing_res,
                                    match_input_res=match_input_res,
                                    domain=domain,
                                    color_map=color_map,
                                    show_progress_bar=True,
                                    )

                    depth_pred: np.ndarray = pipe_out.depth_np
                    depth_colored: Image.Image = pipe_out.depth_colored
                    normal_pred: np.ndarray = pipe_out.normal_np
                    normal_colored: Image.Image = pipe_out.normal_colored

                    # Save as npy
                    rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]

                    depth_npy_save_path = os.path.join(root, f"{rgb_name_base}_depth.npy")
                    if os.path.exists(depth_npy_save_path):
                        logging.warning(f"Existing file: '{depth_npy_save_path}' will be overwritten")
                    np.save(depth_npy_save_path, depth_pred)

                    normal_npy_save_path = os.path.join(root, f"{rgb_name_base}_normal.npy")
                    if os.path.exists(normal_npy_save_path):
                        logging.warning(f"Existing file: '{normal_npy_save_path}' will be overwritten")
                    np.save(normal_npy_save_path, normal_pred)

                    # Colorize and save
                    depth_colored_save_path = os.path.join(root, f"{rgb_name_base}_depth_colored.png")
                    if os.path.exists(depth_colored_save_path):
                        logging.warning(
                            f"Existing file: '{depth_colored_save_path}' will be overwritten"
                        )
                    depth_colored.save(depth_colored_save_path)

                    normal_colored_save_path = os.path.join(root, f"{rgb_name_base}_normal_colored.png")
                    if os.path.exists(normal_colored_save_path):
                        logging.warning(
                            f"Existing file: '{normal_colored_save_path}' will be overwritten"
                        )
                    normal_colored.save(normal_colored_save_path)

    logging.info("Depth and Normal estimation completed.")
