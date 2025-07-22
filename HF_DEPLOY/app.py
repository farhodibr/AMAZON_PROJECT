import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from huggingface_hub import hf_hub_download
from sklearn.neighbors import NearestNeighbors
import pandas as pd

#load precomputed index
archive = hf_hub_download(
    repo_id="glavvrach79/my-recsys-data",
    repo_type="dataset",
    filename="image_index.npz"
)
data       = np.load(archive, allow_pickle=True)
embeddings = data["embeddings"]
item_ids   = data["ids"].tolist()

# ————————————————————————————————
# l oad slim metadata
meta_parquet = hf_hub_download(
    repo_id="glavvrach79/my-recsys-data",
    repo_type="dataset",
    filename="meta_small.parquet"
)
meta_df = pd.read_parquet(meta_parquet)

title_map = meta_df.set_index("parent_asin")["title"].to_dict()
url_map   = meta_df.set_index("parent_asin")["image_url"].to_dict()

# build k‑NN index
knn = NearestNeighbors(n_neighbors=6, metric="cosine")
knn.fit(embeddings)

# prepare ResNet50 extractor
weights    = ResNet50_Weights.IMAGENET1K_V2
base_model = resnet50(weights=weights)
model      = torch.nn.Sequential(*list(base_model.children())[:-1]).eval()
preprocess = weights.transforms()

# recommender
def recommend(img: Image.Image, top_k: float = 5):
    k = int(top_k)
    # extract features
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = model(x).squeeze().numpy()
    # find neighbors + distances
    distances, indices = knn.kneighbors([feat], n_neighbors=k+1, return_distance=True)
    dists, idxs = distances[0], indices[0]
    # collect markdown entries, skipping idx 0 
    md = "## Top Recommendations\n\n"
    for rank, (dist, idx) in enumerate(zip(dists[1:], idxs[1:]), start=1):
        asin = item_ids[idx]
        img_url = url_map.get(asin)
        title   = title_map.get(asin, "No title")
        score   = 1 - dist  # cosine similarity
        amazon_link = f"https://www.amazon.com/dp/{asin}"
        # build a markdown block per item
        md += f"**{rank}. [{title}]({amazon_link})**  \n"
        md += f"Similarity: **{score:.2f}**  \n\n"
        md += f"![{title}]({img_url})  \n\n---\n\n"
    return md

# output in HF
iface = gr.Interface(
    fn=recommend,
    inputs=[
        gr.Image(type="pil", label="Upload your product image"),
        gr.Slider(1, 10, value=5, step=1, label="How many recommendations?")
    ],
    outputs=gr.Markdown(),
    title="DATA 612 RECOMMENDATIONS BY THE IMAGE MODEL",
    description=(
        "Upload a product image and get the top‑k most similar items, "
        "with similarity scores and direct links to Amazon."
    )
)

if __name__ == "__main__":
    iface.launch()
