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
from sklearn.decomposition import PCA

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_images_from_folder(folder, transform, max_images=1500):
    
    already_load = []
    images = []
    filenames = []
    count_i=0
    for filename in os.listdir(folder):
        img_id = int(filename.split(".")[0].replace("ISIC_","").split("_")[0])
        if img_id in already_load or count_i%3==0:
            count_i += 1
            continue
        already_load.append(img_id)
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


def plot_tsne(real_features, generated_features, real_filenames, generated_filenames, output_path):
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
    all_features = np.vstack((real_features, generated_features))
    pca = PCA(n_components=20, random_state=SEED)
    features_pca = pca.fit_transform(all_features)

    tsne_results = tsne.fit_transform(features_pca)

    real_tsne = tsne_results[:len(real_features)]
    generated_tsne = tsne_results[len(real_features):]

    key_index = random.randint(0, len(real_tsne) - 1)
    closest_index = np.argmin(np.linalg.norm(generated_tsne - real_tsne[key_index], axis=1))

    plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 2})
    plt.figure(figsize=(10, 8))
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], c='blue', alpha=0.25)
    plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], c='red', alpha=0.25)


    plt.scatter(real_tsne[key_index, 0], real_tsne[key_index, 1], c='blue', s=150,edgecolors='gray',label='Real Images')
    plt.scatter(generated_tsne[closest_index, 0], generated_tsne[closest_index, 1], c='red', s=150,edgecolors='gray', label='Synthetic Images')

    plt.xticks([])
    plt.yticks([])

    plt.legend( loc='upper right')

    plt.title("(b) Real Versus Synthetic Images in MEL", fontsize=27, pad=20)
    plt.savefig(output_path)


def main():
    parser = argparse.ArgumentParser(description="t-SNE Visualization for Real and Synthetic Images")
    parser.add_argument("--input-image-root-path", type=str, required=True,
                        help="Path to the input image root directory")
    parser.add_argument("--generate-image-root-path", type=str, required=True,
                        help="Path to the generated image root directory")
    parser.add_argument("--output-figure-path", type=str, required=True, help="Path to save the figure")
    parser.add_argument("--class-name", type=str, required=True, help="Class to compare (e.g., mel, akiec, etc.)")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_images_path = os.path.join(args.input_image_root_path, args.class_name, "HAM10000_img_class")
    generated_images_path = os.path.join(args.generate_image_root_path, args.class_name)

    generated_images, generated_filenames = load_images_from_folder(generated_images_path, transform)
    real_images, real_filenames = load_images_from_folder(real_images_path, transform,max_images=len(generated_images) * 3)

    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    real_features = extract_features(model, real_images).squeeze()
    generated_features = extract_features(model, generated_images).squeeze()

    plot_tsne(real_features, generated_features, real_filenames, generated_filenames, args.output_figure_path)


if __name__ == "__main__":
    main()