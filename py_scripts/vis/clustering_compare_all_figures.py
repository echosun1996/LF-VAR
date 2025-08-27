import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import argparse
import seaborn as sns
from tqdm import tqdm

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
fid_scores = {
    'Ours':[0.60, 1.04, 0.69, 0.63, 0.45, 0.67, 1.09, 0.74],
    'VAR':[0.74, 1.58, 0.44, 0.76, 0.44, 0.56, 1.04, 0.79],
    'Derm-T2IM':[6.52, 6.02, 3.32, 5.31, 4.74, 6.26, 5.69, 5.41],
    'Diffusion':[1.22, 1.47, 0.78, 1.15, 0.90, 1.57, 0.64, 1.11]
}

def load_images_from_folder(folder, transform, max_images=200):
    
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            images.append(transform(image))
            filenames.append(filename)
            if len(images) >= max_images:
                break
        except:
            continue
    return images, filenames


def extract_features(model, images):
    
    model.eval()
    with torch.no_grad():
        images = torch.stack(images).to("cuda" if torch.cuda.is_available() else "cpu")
        features = model(images)
    return features.cpu().numpy()


def compute_feature_distances(real_features, generated_features):
    
    distances = np.linalg.norm(real_features[:, None, :] - generated_features[None, :, :], axis=2)
    mean_distances = distances.mean(axis=1)
    return mean_distances


def plot_trends(classes, dataset_names, mean_distances, std_distances, fid_scores, output_path):
    
    classes.append("Overall")
    plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 2})
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()

    handles = []
    labels = []

    for dataset_name in dataset_names:
        if dataset_name == "Center":
            continue


        mean_values = mean_distances[dataset_name]
        std_values = std_distances[dataset_name]

        overall_mean = np.mean(mean_values)
        overall_std = np.mean(std_values)
        mean_values.append(overall_mean)
        std_values.append(overall_std)

        color = next(ax1._get_lines.prop_cycler)['color']

        texts = []
        for i, fid in enumerate(fid_scores[dataset_name]):
            k = ax1.scatter(i, fid, marker='x', c=color, s=50)
            spec = 0
            if dataset_name == "Ours":
                if i==0:
                    spec=5
                if i ==2 or i==3:
                    spec=5
                if i ==4:
                    spec=-15
                if i ==6:
                    spec=-5
                if i ==7:
                    spec=12
            elif dataset_name == "VAR":
                if i == 1:
                    spec=-10
            spec_off = spec*0.01 * np.ptp(fid_scores[dataset_name])
            y_offset = random.randint(4,10) *0.01 * np.ptp(fid_scores[dataset_name])
            adjusted_y = fid -  y_offset -spec_off

            tex = ax1.text(i + 0.1, adjusted_y, f'{fid:.2f}', ha='left', va='center', fontsize=10, color=color)


        line, = ax1.plot(classes, mean_values, marker='o', label=dataset_name, color=color)
        ax1.fill_between(classes, np.array(mean_values) - np.array(std_values),
                         np.array(mean_values) + np.array(std_values), alpha=0.15, color=color)

        handles.append(line)
        labels.append(dataset_name)



    ax1.set_xlim(-0.5, len(classes) - 0.5)
    ax1.set_ylim(0,7)
    ax2.set_ylim(0,7)
    ax1.set_xlabel("Skin Disease Class")
    ax1.set_ylabel("Feature Distance")
    ax2.set_ylabel("FID Score")
    ax1.set_title("(c) Feature Distance and FID across Lesions", fontsize=27, pad=20)
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels([cls.upper() if i < len(classes) - 1 else cls for i, cls in enumerate(classes)], rotation=45)
    ax1.grid()
    plt.draw()
    fid_marker = ax2.scatter([], [], marker='x', color='black', label='FID Score')
    handles.append(fid_marker)
    labels.append('FID Score')
    leg = ax1.legend(handles, labels, loc='upper right')
    fig.canvas.draw()
    plt.savefig(output_path,bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser(description="t-SNE and Feature Distance Analysis for Multiple Datasets")
    parser.add_argument("--input-image-root-path", type=str, required=True,
                        help="Path to the input image root directory")
    parser.add_argument("--generate-image-root-path", type=str, nargs='+', required=True,
                        help="Paths to the generated image root directories")
    parser.add_argument("--model-names", type=str, nargs='+', required=True,
                        help="Names of the models corresponding to each generated image path")
    parser.add_argument("--output-figure-path", type=str, required=True, help="Path to save the figure")
    args = parser.parse_args()

    if len(args.generate_image_root_path) != len(args.model_names):
        raise ValueError("The number of model names must match the number of generate image root paths.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    classes = sorted(os.listdir(args.input_image_root_path))
    mean_distances = {model_name: [] for model_name in args.model_names}
    std_distances = {model_name: [] for model_name in args.model_names}
    distances = {model_name: [] for model_name in args.model_names}


    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    for class_name in tqdm(classes, desc="Processing classes"):
        real_images_path = os.path.join(args.input_image_root_path, class_name, "HAM10000_img_class")

        if not os.path.exists(real_images_path):
            continue

        real_images, _ = load_images_from_folder(real_images_path, transform, max_images=500)
        if not real_images:
            continue

        real_features = extract_features(model, real_images).squeeze()

        for generate_path, model_name in zip(args.generate_image_root_path, args.model_names):
            generated_images_path = os.path.join(generate_path, class_name)
            if os.path.exists(os.path.join(generated_images_path, "HAM10000_img_class")):
                generated_images_path = os.path.join(generated_images_path, "HAM10000_img_class")

            if not os.path.exists(generated_images_path):
                print(f"Generated Images Folder doesn't exist: {generated_images_path}")
                exit(-1)

            generated_images, _ = load_images_from_folder(generated_images_path, transform)
            if not generated_images:
                continue

            generated_features = extract_features(model, generated_images).squeeze()
            distances_temp = compute_feature_distances(real_features, generated_features)
            distances[model_name] = distances_temp

        for model_name, model_distances in distances.items():
            model_distances = np.abs(np.array(model_distances) - np.array(distances["Center"]))
            mean_distances[model_name].append(model_distances.mean())
            std_distances[model_name].append(model_distances.std())

    plot_trends(classes, args.model_names, mean_distances, std_distances, fid_scores, args.output_figure_path)


if __name__ == "__main__":
    main()