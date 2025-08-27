import os
import argparse
import pandas as pd
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict


def main(args):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        os.path.join(args.model_dir,"models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590"),
        torch_dtype=torch.float16
    )
    pipe.to("cuda")

    metadata = pd.read_csv(args.metadata_file)

    test_metadata = metadata[metadata['dataset_split'] == 'test']

    count_dict = defaultdict(int)

    for index, row in tqdm(test_metadata.iterrows(), total=len(test_metadata)):
        img_path = row["img_path"]
        seg_path = row["seg_path"]
        prompt = row["prompt"]
        lbl = row["class"]

        if count_dict[lbl] >= 1500:
            continue

        folder_path = os.path.join(args.output, lbl)
        os.makedirs(folder_path, exist_ok=True)

        image = Image.open(os.path.join(args.data_root,img_path)).convert("RGB").resize((512, 512))
        mask = Image.open(os.path.join(args.data_root,seg_path)).convert("L").resize((512, 512))

        for _ in range(300):
            if count_dict[lbl] >= 1500:
                break
            result = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
            result = result.resize((512, 512))

            save_path = os.path.join(folder_path, f"{str(count_dict.get(lbl)).zfill(5)}.png")
            result.save(save_path)

            count_dict[lbl] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Local save path of the model")
    parser.add_argument("--data_root", required=True, type=str, help="Data root path")
    parser.add_argument("--metadata_file", required=True, type=str, help="Path to the metadata file")
    parser.add_argument("--output", required=True, type=str, help="Directory to save output images")
    args = parser.parse_args()

    main(args)