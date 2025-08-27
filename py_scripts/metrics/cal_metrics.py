import argparse
import torch
import os
import numpy as np
from torch_fidelity import calculate_metrics
from pytorch_fid import fid_score
from PIL import Image
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim
import glob

def load_images_from_directory(directory, transform=None):
    
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for format in supported_formats:
        image_paths.extend(glob.glob(os.path.join(directory, format)))
        image_paths.extend(glob.glob(os.path.join(directory, format.upper())))
    
    if not image_paths:
        raise ValueError(f"No images found in directory: {directory}")
    
    images = []
    for path in sorted(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            if transform:
                img = transform(img)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {path}: {e}")
    
    if not images:
        raise ValueError(f"No valid images could be loaded from directory: {directory}")
    
    return torch.stack(images)

def compute_ms_ssim(dir1, dir2, device='cuda'):
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    images1 = load_images_from_directory(dir1, transform).to(device)
    images2 = load_images_from_directory(dir2, transform).to(device)
    
    min_count = min(len(images1), len(images2))
    if len(images1) != len(images2):
        print(f"Warning: Different number of images. Using first {min_count} from each directory.")
        images1 = images1[:min_count]
        images2 = images2[:min_count]
    
    ms_ssim_values = []
    batch_size = 10
    
    for i in range(0, len(images1), batch_size):
        end_idx = min(i + batch_size, len(images1))
        batch1 = images1[i:end_idx]
        batch2 = images2[i:end_idx]
        
        ms_ssim_val = ms_ssim(batch1, batch2, data_range=1.0, size_average=True)
        ms_ssim_values.append(ms_ssim_val.item())
    
    return np.mean(ms_ssim_values)

def main():
    parser = argparse.ArgumentParser(description="Compute IS, FID, and MS-SSIM metrics")
    parser.add_argument("--input1", type=str, required=True,
                        help="Path to the first input (generated images directory)")
    parser.add_argument("--input2", type=str, required=True,
                        help="Path to the second input (original images directory)")
    parser.add_argument("--output", type=str, default="results.txt",
                        help="Path to save the computed metrics (default: results.txt)")
    parser.add_argument("--gpu", action="store_true", default=True, 
                        help="Use GPU for computation (default: True)")
    parser.add_argument("--isc", action="store_true", default=True, 
                        help="Compute Inception Score (IS)")
    parser.add_argument("--fid", action="store_true", default=True, 
                        help="Compute Frechet Inception Distance (FID)")
    parser.add_argument("--ms_ssim", action="store_true", default=True, 
                        help="Compute Multi-Scale SSIM (MS-SSIM)")

    args = parser.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    results = {}

    if args.isc:
        print("Computing Inception Score...")
        try:
            metrics = calculate_metrics(
                input1=args.input1,
                input2=args.input2,
                cuda=(device == 'cuda'),
                isc=True,
            )
            results.update(metrics)
        except Exception as e:
            print(f"Error computing IS: {e}")

    if args.fid:
        print("Computing FID...")
        try:
            fid = fid_score.calculate_fid_given_paths(
                [args.input1, args.input2],
                batch_size=50,
                device=device,
                dims=2048
            )
            results['FID'] = fid
        except Exception as e:
            print(f"Error computing FID: {e}")

    if args.ms_ssim:
        print("Computing MS-SSIM...")
        try:
            ms_ssim_value = compute_ms_ssim(args.input1, args.input2, device)
            results['MS_SSIM'] = ms_ssim_value
        except Exception as e:
            print(f"Error computing MS-SSIM: {e}")

    with open(args.output, "w") as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
            print(f"{metric}: {value}")

    print(f"Metrics saved to {args.output}")

if __name__ == "__main__":
    main()