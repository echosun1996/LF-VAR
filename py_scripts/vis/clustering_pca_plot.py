import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Figures and Fix Radiomics Features Generation")
parser.add_argument("--input-file-path", type=str, required=True,
                    help="Path to the input csv")
parser.add_argument("--output-figure-path", type=str, required=True,
                    help="Path to save the clustering figures")
parser.add_argument("--output-csv-path", type=str, required=True,
                    help="Path to save the fixed radiomics features")
args = parser.parse_args()

file_path = args.input_file_path
df = pd.read_csv(file_path)

feature_columns = [col for col in df.columns if col.startswith("original_")]
X = df[feature_columns]
y = df["category"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=30, random_state=42)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=100, learning_rate=500, early_exaggeration=5.0, metric='cosine', init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_pca)

plt.figure(figsize=(10, 7))
unique_categories = y.unique()
colors = plt.cm.tab10.colors[:len(unique_categories)]

for i, category in enumerate(unique_categories):
    indices = (y == category)
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
                color=colors[i], label=f"Category {category}", alpha=0.6, s=8)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("2D t-SNE of Radiomics Features")
plt.legend()
plt.savefig(args.output_figure_path)

output_file = args.output_csv_path
df.groupby("category")[feature_columns].mean().to_csv(output_file)

print(f"Saved grouped features by category to: {output_file}")