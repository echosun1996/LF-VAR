import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser(description="Figures and Fix Radiomics Features Generation")
parser.add_argument("--input-image-root-path", type=str, required=True,
                    help="Path to the input image root directory")
parser.add_argument("--output-figure-path", type=str, required=True,
                    help="Path to save the clustering figures")
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label in self.classes:
            class_path = os.path.join(root_dir, label, 'HAM10000_img_class')
            if os.path.exists(class_path):
                images = [os.path.join(class_path, img) for img in os.listdir(class_path) if
                          img.endswith(('.png', '.jpg', '.jpeg'))]
                self.image_paths.extend(images)
                self.labels.extend([label.upper()] * len(images))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


dataset = ImageDataset(args.input_image_root_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device)
model.eval()

features = []
labels = []
with torch.no_grad():
    for images, lbls in dataloader:
        images = images.to(device)
        output = model(images)
        output = output.view(output.size(0), -1)
        features.append(output.cpu().numpy())
        labels.extend(lbls)

features = np.vstack(features)

pca = PCA(n_components=100, random_state=SEED)
features_pca = pca.fit_transform(features)
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
features_tsne = tsne.fit_transform(features_pca)

plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 2})
plt.figure(figsize=(10, 8))


unique_labels = sorted(set(labels))

base_cmap = plt.cm.get_cmap('tab10', len(unique_labels))
colors = {label: mcolors.to_rgba(base_cmap(i)) for i, label in enumerate(unique_labels)}



class_representatives = {}
for i, label in enumerate(unique_labels):
    indices = [j for j, lbl in enumerate(labels) if lbl == label]
    plt.scatter(features_tsne[indices, 0], features_tsne[indices, 1],s=50, label=None, color=colors[label], alpha=0.25)

    class_representatives[label] = (features_tsne[indices[0], 0], features_tsne[indices[0], 1], colors[label])

for label, (x, y, color) in class_representatives.items():
    plt.scatter(x, y, color=color, edgecolors='gray', s=200, zorder=3, label=label)

plt.legend(markerscale=0.5, loc='upper right')

plt.xticks([])
plt.yticks([])

plt.title("(a) Visualization of Image Features", fontsize=27, pad=20)

plt.savefig(args.output_figure_path)
plt.show()