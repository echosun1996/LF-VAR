import os
import csv
import argparse
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

def replace_dx(input):
    dx_map = {
        "akiec": "actinic keratoses",
        "bcc": "basal cell carcinoma",
        "bkl": "benign keratosis",
        "df": "dermatofibroma",
        "mel": "melanoma",
        "nv": "melanocytic nevi",
        "vasc": "vascular lesions"
    }
    try:
        return dx_map[input]
    except KeyError:
        print(f"Error 'dx': [{input}] in meta file not match.")
        exit(-1)

def main(args):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    os.makedirs(args.output, exist_ok=True)

    metadata = {}
    with open(args.meta, mode="r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metadata[row["image_id"]] = replace_dx(row["dx"])

    for img_filename in os.listdir(args.image):
        if img_filename.endswith((".png", ".jpg", ".jpeg")):
            base_name = os.path.splitext(img_filename)[0]
            img_file = os.path.join(args.image, img_filename)
            seg_file = os.path.join(args.seg, f"{base_name}_segmentation.png")

            if not os.path.exists(seg_file):
                print(f"Segmentation mask not found for {img_filename}, skipping.")
                exit(-1)

            prompt = metadata.get(base_name)
            if prompt is None:
                print(f"Metadata for {base_name} not found, skipping.")
                exit(-1)

            image = Image.open(img_file).convert("RGB")
            mask_image = Image.open(seg_file).convert("L")

            try:
                result = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
                output_file = os.path.join(args.output, f"{base_name}.png")
                result.save(output_file)
                print(f"Saved result to {output_file}")
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")
                exit(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting with metadata.")
    parser.add_argument("--image", required=True, help="Path to the folder containing images.")
    parser.add_argument("--seg", required=True, help="Path to the folder containing segmentation masks.")
    parser.add_argument("--meta", required=True, help="Path to the metadata CSV file.")
    parser.add_argument("--output", required=True, help="Path to the output folder.")
    args = parser.parse_args()
    main(args)