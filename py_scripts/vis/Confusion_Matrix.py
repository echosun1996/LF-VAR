import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


data = 



pairs = {}
for line in data.strip().split("\n"):
    key, value = line.split(":")
    pairs[key] = float(value)

categories = sorted(set([pair.split("_")[0] for pair in pairs.keys()] + [pair.split("_")[1] for pair in pairs.keys()]))

matrix = np.zeros((len(categories), len(categories)))
categories_upper = [cat.upper() for cat in categories]

for key, value in pairs.items():
    row, col = key.split("_")
    i, j = categories.index(row), categories.index(col)
    matrix[i, j] = value

conf_matrix = pd.DataFrame(matrix, index=categories, columns=categories)
conf_matrix_upper = pd.DataFrame(matrix, index=categories_upper, columns=categories_upper)

plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 2})
plt.figure(figsize=(10, 8))
ax=sns.heatmap(conf_matrix_upper, annot=True, fmt=".2f", cmap="Blues", xticklabels=categories_upper, yticklabels=categories_upper,
            linewidths=0.5, linecolor='white')

plt.xlabel("Generated Skin Lesion Type", fontsize=20, labelpad=15)
plt.ylabel("Reference Skin Lesion Type", fontsize=20, labelpad=15)
plt.title("(a) Inter-class Results for Seven Skin Infections \n\n(b) FID Confusion Matrix for Inter-class Synthesis\n", fontsize=30)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("conf_matrix.png",bbox_inches='tight')