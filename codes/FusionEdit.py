from diffusers import FluxPipeline
import argparse
import random
import numpy as np
from FusionEdit_utils import *
from PIL import Image
import torch
from torchvision import transforms
import os


def main(args):
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained FLUX model
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    scheduler = pipe.scheduler

    # Load and preprocess image
    img = Image.open(args.image_path)
    preprocess = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = preprocess(img).unsqueeze(0)

    # Prepare a single "dataloader" for processing
    dataloader = [[img_tensor, args.source_prompt, args.target_prompt]]

    for img_input, source_prompt, target_prompt in dataloader:
        # Hyperparameters
        T_steps = 28
        src_guidance_scale = 1.5
        tar_guidance_scale = 5.5
        n_min = 0
        n_max = 24
        seed = 10

        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Encode source prompt and preprocess image
        src_prompt = source_prompt
        tar_prompt = target_prompt
        image_src = pipe.image_processor.preprocess(img_input).to(device).half()
        
        with torch.autocast("cuda"), torch.inference_mode():
            x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode().to(device)

        # Normalize latent
        x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        x0_src = x0_src.to(device)

        # Run FusionEdit
        x0_tar = FusionEdit(
            pipe,
            scheduler,
            x0_src,
            src_prompt,
            tar_prompt,
            T_steps,
            src_guidance_scale,
            tar_guidance_scale,
            n_min,
            n_max
        )

        # Decode edited latent back to image
        x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        with torch.autocast("cuda"), torch.inference_mode():
            image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]

        # Postprocess and save result
        image_tar = pipe.image_processor.postprocess(image_tar)
        os.makedirs(args.output_path, exist_ok=True)
        image_tar[0].save(os.path.join(args.output_path, "result.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test interpolated denoising with different parameters.")
    parser.add_argument("--model_path", type=str, default="FLUX.1-dev",
                        help="Path to the pretrained FLUX model")
    parser.add_argument("--image_path", type=str,
                        default="./example_images/000000000007.jpg",
                        help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="outputs",
                        help="Folder to save output images")
    parser.add_argument("--source-prompt", type=str,
                        default="a german shepherd dog stands on the grass with mouth closed",
                        help="Description of the source image content")
    parser.add_argument("--target-prompt", type=str,
                        default="a german shepherd dog stands on the grass with mouth opened",
                        help="Description of the desired edited content")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of the output image")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of the output image")
    args = parser.parse_args()
    main(args)
