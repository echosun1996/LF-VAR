import numpy as np
from collections import defaultdict

data_original = 


data = defaultdict(list)

for line in data_original.strip().split("\n"):
    key, value = line.split(":")
    suffix = key.split("_")[-1]
    data[suffix].append(float(value))

data = dict(data)
print(data)

stats = {key: (np.mean(values), np.std(values)) for key, values in data.items()}

for key, (mean, std) in stats.items():
    print(f"{key}: Mean = {mean:.2f}, Std = {std:.2f}")