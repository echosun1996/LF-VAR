import torch
from diffusers import StableDiffusionInpaintPipeline
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, type=str, help="local save path")

args = parser.parse_args()

local_dir = args.model_dir

if not os.path.exists(local_dir):
    os.makedirs(local_dir)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    cache_dir=local_dir
)