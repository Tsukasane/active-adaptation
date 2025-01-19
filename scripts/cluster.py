import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase
from torchrl.modules import ProbabilisticActor
from torchrl.data import CompositeSpec
from dataclasses import dataclass, field

import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import umap.umap_ as umap

num=6

embedding_dir = "/home/ubuntu/Desktop/workspace/active-adaptation/scripts/checkpoints/embeddings"
embedding_files = os.listdir(embedding_dir)[:num]

embeds = None
label = None
for i, embed_file in enumerate(embedding_files):
    path = os.path.join(embedding_dir, embed_file)
    buffer = torch.load(path)[:1].reshape(-1, 64)
    if embeds is None:
        embeds = buffer
        label = torch.ones(buffer.shape[0]) * i
    else:
        embeds = torch.cat((embeds, buffer), 0)
        label = torch.cat((label, torch.ones(buffer.shape[0]) * i), 0)

embeds = embeds.cpu().detach().numpy()
label = label.cpu().detach().numpy()
print(embeds.shape, label.shape)

scaler = StandardScaler()
embeds = scaler.fit_transform(embeds)

pca = PCA(n_components=50, random_state=3)
embeds = pca.fit_transform(embeds)

umap_model = umap.UMAP(
    n_neighbors=200,
    min_dist=0.99,
    n_components=2,
    random_state=3
)
umap_results = umap_model.fit_transform(embeds)

df = pd.DataFrame(
    {
        "x": umap_results[:,0],
        "y": umap_results[:,1],
        "label": [embedding_files[int(l)].split(".")[0] for l in label]
    }
)

my_colors = [
    "#e6194B",  # Red
    "#3cb44b",  # Green
    "#ffe119",  # Yellow
    "#4363d8",  # Blue
    "#f58231",  # Orange
    "#911eb4",  # Purple
    "#46f0f0",  # Cyan
    "#f032e6",  # Magenta
    "#bcf60c",  # Lime
    "#fabebe",  # Light Pink
    "#008080",  # Teal
    "#e6beff",  # Lavender
    "#9A6324",  # Brown
][:num]

plt.figure(figsize=(16,10))
sns.scatterplot(
    data=df,
    x="x",
    y="y",
    hue="label",
    palette=my_colors,
    legend="full",
    s=10,
    alpha=0.6
)
plt.title("Projection of Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()