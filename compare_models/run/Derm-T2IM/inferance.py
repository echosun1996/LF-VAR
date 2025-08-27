import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Generate images using DiffusionPipeline based on input prompts.")
    parser.add_argument("--prompt", required=True, type=str, help="The text prompt for the model.")
    parser.add_argument("--output", required=True, type=str, help="Output directory for generated images.")
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of batch size.")
    parser.add_argument("--pretrain", required=True, type=str, help="Path to the pre-trained model weights.")

    args = parser.parse_args()

    prompt = args.prompt
    output_dir = Path(args.output)
    num_images = args.n
    pretrain_path = args.pretrain
    batch_size = args.batch_size


    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.pretrain).exists():
        raise FileNotFoundError(f"Pre-trained model directory not found at {args.pretrain}")

    print("Loading model...")
    print(pretrain_path)
    pipe = StableDiffusionPipeline.from_single_file(pretrain_path,  torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    print(f"Generating {num_images} image(s) for prompt: '{prompt}'")
    image_size = (512, 512)
    for i in tqdm(range(0, num_images, batch_size)):
        current_batch_size = min(batch_size, num_images - i)
        batch_images = pipe([prompt] * current_batch_size).images

        for j, image in enumerate(batch_images):
            output_index = i + j
            output_filename = f"{str(output_index).zfill(5)}.png"
            output_path = output_dir / output_filename
            resized_image = image.resize(image_size, Image.LANCZOS)
            resized_image.save(output_path)

if __name__ == "__main__":
    main()